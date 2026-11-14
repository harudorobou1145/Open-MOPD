#!/usr/bin/env python3
"""WikiText-2 测试集困惑度 (Perplexity, PPL) 统一评估脚本。

功能说明:
    采用标准的不重叠 1024-token 分块协议 (忽略每块首个无前文 token，保留末尾不足块，
    FP32 累加交叉熵，共评分 296,902 个 token)，对以下三类模型在同一测试集上进行严格对齐评估:
      1) AnyPrecision 共享码本学生模型训练前 (before) 与训练后 (after) 各精度位宽 (如 W2/W4/W8)
      2) FSDP 多卡分片目录 checkpoint (自动通过 torchrun 聚合并由 rank 0 执行评测) 或单文件 .pt
      3) 独立的量化或全精度基线/教师模型 (--standalone-model)

启动示例 (在 /workspace/my_project/Open-MOPD 目录下执行):
    1) 单卡评估 FSDP NO_SHARD 训练后的单文件权重 (以 rank_0.pt 为例，同时评测 W2/W4/W8):
       HF_HOME=/cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       CUDA_VISIBLE_DEVICES=0 /usr/bin/python my_scripts/eval_anyprecision_wikitext.py \
           --cache "/data/anyprecision_w2w8/cache/quantized/(Qwen3.5-0.8B)-w8_orig2-gc1-c4_s100_blk512" \
           --checkpoint /data/anyprecision_mopd/parent8_w4_only_n2_vllm_full_20261113_final/checkpoints/global_step_1485/actor/model_world_size_2_rank_0.pt \
           --model-precisions 2 3 4 5 6 7 8 \
           --precisions 2 4 8 \
           --output /data/anyprecision_mopd/eval_w4_only_ppl

    2) 多卡聚合 FSDP 分片目录 checkpoint 并评测 (nproc_per_node 需等于训练时的 world_size):
       HF_HOME=/cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       CUDA_VISIBLE_DEVICES=0,1 /usr/bin/python -m torch.distributed.run --nproc_per_node=2 \
           my_scripts/eval_anyprecision_wikitext.py \
           --cache "/data/anyprecision_w2w8/cache/quantized/(Qwen3.5-0.8B)-w8_orig2-gc1-c4_s100_blk512" \
           --checkpoint /data/anyprecision_mopd/w4w8_run/checkpoints/global_step_1485/actor \
           --model-precisions 2 3 4 5 6 7 8 \
           --precisions 2 4 8 \
           --output /data/anyprecision_mopd/eval_fsdp_ppl

    3) 独立评测单个 AutoRound 量化教师模型或 BF16 基座模型:
       HF_HOME=/cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       CUDA_VISIBLE_DEVICES=0 /usr/bin/python my_scripts/eval_anyprecision_wikitext.py \
           --standalone-model /data/models/Qwen3.5-0.8B-int4-AutoRound \
           --quant-backend auto_round:torch_zp \
           --output /data/anyprecision_mopd/eval_w4_teacher_ppl

    4) 快速冒烟测试 (仅评估前 4 个文本块):
       CUDA_VISIBLE_DEVICES=0 /usr/bin/python my_scripts/eval_anyprecision_wikitext.py \
           --max-blocks 4 --output /tmp/eval_smoke_test
"""

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as functional


def load_checkpoint(checkpoint):
    """加载模型权重：支持单个 .pt 文件，或通过分布式进程组聚合 FSDP 分片目录。"""
    path = Path(checkpoint)
    # 若传入的是单个权重文件（例如 NO_SHARD 模式下的 rank_0.pt），直接在 CPU 上映射读取
    if not path.is_dir():
        return torch.load(path, map_location='cpu', weights_only=False, mmap=True)

    # 若传入的是 FSDP actor 目录，则初始化 NCCL 并将 DTensor 还原为完整张量汇总到 rank 0
    import torch.distributed as distributed
    from torch.distributed.tensor import DTensor

    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    distributed.init_process_group('nccl')
    rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    state = torch.load(path / f'model_world_size_{world_size}_rank_{rank}.pt',
                       map_location=f'cuda:{torch.cuda.current_device()}', weights_only=False)
    restored = {}
    for name in sorted(state):
        value = state.pop(name)
        full = value.full_tensor() if isinstance(value, DTensor) else value
        if rank == 0:
            restored[name] = full.cpu()
        del value, full
    distributed.barrier()
    distributed.destroy_process_group()
    torch.cuda.empty_cache()
    # 非主进程返回 None，外层主函数据此直接退出，仅由 rank 0 继续执行单卡 PPL 计算
    return restored if rank == 0 else None


def main():
    parser = argparse.ArgumentParser(description='Compare pre/post MOPD WikiText-2 test perplexity.')
    parser.add_argument('--model', default='/workspace/my_project/models/Qwen3.5-0.8B',
                        help='基座学生模型目录路径')
    parser.add_argument('--anyprecision-repo', default='/workspace/my_project/any-precision-llm',
                        help='any-precision-llm 代码库路径（用于导入 TrainableAnyPrecisionLinear）')
    parser.add_argument('--cache', default=(
        '/workspace/my_project/outputs/anyprecision_qwen35_08b/mlp_seed4_parent8_s0/'
        'calibration_cache/quantized/(Qwen3.5-0.8B)-w8_orig4-gc1-c4_s100_blk512'),
        help='AnyPrecision 共享码本校准缓存目录')
    parser.add_argument('--checkpoint', default=(
        '/data/anyprecision_mopd/batched_full_20261105_085300/checkpoints/'
        'global_step_1485/actor/model_world_size_1_rank_0.pt'),
        help='训练后保存的 checkpoint 文件或 FSDP actor 目录')
    parser.add_argument('--output', required=True,
                        help='评估结果与 manifest.json 输出目录（不可覆盖已有结果）')
    parser.add_argument('--sequence-length', type=int, default=1024,
                        help='不重叠滑窗评估块长度，默认为 1024')
    parser.add_argument('--precisions', type=int, nargs='+', default=[4, 8],
                        help='需要依次评测的学生位宽列表，例如 2 4 8')
    parser.add_argument('--model-precisions', type=int, nargs='+',
                        help='checkpoint 中安装的完整位宽集合（如 2 3 4 5 6 7 8），用于严格匹配 state_dict')
    parser.add_argument('--max-blocks', type=int,
                        help='仅评测前 N 个块（用于快速冒烟验证）')
    parser.add_argument('--standalone-model',
                        help='若指定该路径，则跳过 AnyPrecision 学生改造，直接评测该独立模型')
    parser.add_argument('--quant-backend',
                        help='独立量化模型的可选后端覆盖（如 auto_round:torch_zp 或 torch_awq）')
    parser.add_argument('--include-teachers', action='store_true',
                        help='在同一次运行中附带评测默认 W4/W8 教师模型（仅限 --precisions 4 8）')
    parser.add_argument('--teacher-w4', default='/workspace/my_project/models/Qwen3.5-0.8B-W4A16-AutoRound-AWQ',
                        help='--include-teachers 模式下的 W4 教师路径')
    parser.add_argument('--teacher-w8', default='/workspace/my_project/models/Qwen3.5-0.8B-GPTQ-MLP-W8-cal128-seq2048',
                        help='--include-teachers 模式下的 W8 教师路径')
    args = parser.parse_args()

    if args.include_teachers and args.precisions != [4, 8]:
        parser.error('--include-teachers currently requires --precisions 4 8')
    if args.sequence_length < 2 or (args.max_blocks is not None and args.max_blocks < 1):
        parser.error('sequence-length >= 2 and max-blocks >= 1 required')

    # 若 checkpoint 是分布式目录，先由所有 rank 协同还原完整权重，非 0 号进程完成后直接退出
    checkpoint_state = None
    if not args.standalone_model and Path(args.checkpoint).is_dir():
        checkpoint_state = load_checkpoint(args.checkpoint)
        if checkpoint_state is None:
            return

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'results.json').exists():
        raise FileExistsError('Use a new output directory to avoid overwriting results')

    sys.path.insert(0, args.anyprecision_repo)
    from any_precision.trainable import install_trainable_qwen, student_precision
    from datasets import load_dataset
    from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForConditionalGeneration

    # 加载分词器与官方 WikiText-2 test 集，按双换行拼接全文后切分为不重叠的固定长度 token 块
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')
    text = '\n\n'.join(dataset['text'])
    tokens = tokenizer(text, return_tensors='pt', add_special_tokens=False)['input_ids']
    blocks = [tokens[:, start:start + args.sequence_length]
              for start in range(0, tokens.shape[1], args.sequence_length)
              if tokens.shape[1] - start >= 2]
    if args.max_blocks is not None:
        blocks = blocks[:args.max_blocks]
    if not blocks:
        raise ValueError('No scorable tokens')

    # 记录评测元数据清单，便于审计数据集指纹、评分 token 数与模型路径
    manifest = dict(dataset='Salesforce/wikitext', config='wikitext-2-raw-v1', split='test',
                    dataset_fingerprint=dataset._fingerprint,
                    text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    source_tokens=tokens.numel(), scored_tokens=sum(b.numel() - 1 for b in blocks),
                    blocks=len(blocks), sequence_length=args.sequence_length,
                    protocol='Non-overlapping blocks, include final partial block, omit each block first token; '
                             'no chat template or added special tokens; summed FP32 token cross entropy.',
                    dtype='FP32 model and delta with BF16 autocast', model=args.model,
                    cache=args.cache, checkpoint=args.checkpoint, precisions=args.precisions,
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    smoke_test=args.max_blocks is not None,
                    standalone_model=args.standalone_model,
                    teachers=({'4': args.teacher_w4, '8': args.teacher_w8} if args.include_teachers else {}),
                    teacher_dtype='FP16, matching training; W4 torch_awq, W8 torch',
                    independence_note='Official test split; calibration contamination has not been audited.')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    # 构建待测模型：独立模型直接加载，AnyPrecision 学生则先加载 FP32 基座再替换可训练共享码本层
    if args.standalone_model:
        candidate_config = AutoConfig.from_pretrained(args.standalone_model)
        if args.quant_backend:
            candidate_config.quantization_config['backend'] = args.quant_backend
        manifest.update(dtype='BF16 model with BF16 autocast', model=args.standalone_model,
                        cache=None, checkpoint=None,
                        quantization_config=getattr(candidate_config, 'quantization_config', None))
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        candidate_tokenizer = AutoTokenizer.from_pretrained(args.standalone_model)
        if candidate_tokenizer.get_vocab() != tokenizer.get_vocab():
            raise ValueError('Candidate and reference token vocabularies differ')
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            args.standalone_model, config=candidate_config, dtype=torch.bfloat16,
            device_map={'': 0}, attn_implementation='sdpa')
    else:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            args.model, dtype=torch.float32, attn_implementation='sdpa')
        install_trainable_qwen(model, args.cache, precisions=args.model_precisions or args.precisions)

    model.eval().requires_grad_(False)
    model.to('cuda')

    results = []
    stages = ('before', 'after', 'teacher') if args.include_teachers else ('before', 'after')
    if args.standalone_model:
        stages = ('standalone',)

    # 按阶段（训练前 delta=0 -> 训练后加载 checkpoint -> 可选教师）与位宽逐一计算精确 PPL
    for stage in stages:
        if stage == 'after':
            state = checkpoint_state if checkpoint_state is not None else load_checkpoint(args.checkpoint)
            model.load_state_dict(state, strict=True)
            checkpoint_state = None
            del state
        for bits in ((None,) if args.standalone_model else args.precisions):
            if stage == 'teacher':
                del model
                torch.cuda.empty_cache()
                teacher_path = args.teacher_w4 if bits == 4 else args.teacher_w8
                teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_path)
                if teacher_tokenizer.get_vocab() != tokenizer.get_vocab():
                    raise ValueError('Teacher and student token vocabularies differ')
                if bits == 8:
                    from gptqmodel import GPTQModel

                    model = GPTQModel.load(
                        teacher_path, backend='torch', torch_dtype=torch.float16,
                        device_map={'': 0}, attn_implementation='sdpa').model
                else:
                    config = AutoConfig.from_pretrained(teacher_path)
                    config.quantization_config['backend'] = 'torch_awq'
                    model = Qwen3_5ForConditionalGeneration.from_pretrained(
                        teacher_path, config=config, dtype=torch.float16,
                        device_map={'': 0}, attn_implementation='sdpa')
                model.eval().requires_grad_(False)

            started = time.monotonic()
            total_nll = 0.0
            total_tokens = 0
            # 学生模型通过 student_precision 上下文切换当前前向使用的位宽码本
            precision_context = nullcontext() if stage in ('teacher', 'standalone') else student_precision(model, bits)
            with torch.inference_mode(), precision_context:
                for index, block in enumerate(blocks, 1):
                    inputs = block.to('cuda')
                    with torch.autocast('cuda', dtype=torch.float16 if stage == 'teacher' else torch.bfloat16):
                        logits = model(input_ids=inputs, use_cache=False).logits
                    # 使用 FP32 计算每个预测位置的负对数似然总和，避免半精度累加误差
                    nll = functional.cross_entropy(
                        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                        inputs[:, 1:].reshape(-1), reduction='sum')
                    if not torch.isfinite(nll):
                        raise ValueError(f'Nonfinite NLL: {stage} W{bits} block {index}')
                    total_nll += nll.item()
                    total_tokens += inputs.numel() - 1
                    del logits, nll
                    if index == 1 or index % 20 == 0 or index == len(blocks):
                        print(f'{stage} W{bits}: {index}/{len(blocks)} '
                              f'ppl={math.exp(total_nll / total_tokens):.6f}', flush=True)

            result = dict(stage=stage, bits=bits, nll=total_nll, tokens=total_tokens,
                          ppl=math.exp(total_nll / total_tokens), seconds=time.monotonic() - started)
            results.append(result)
            (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
            print(json.dumps(result), flush=True)

    print(f'COMPLETE: {len(results)} matched evaluation runs saved.', flush=True)


if __name__ == '__main__':
    main()
