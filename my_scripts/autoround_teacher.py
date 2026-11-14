#!/usr/bin/env python3
"""AutoRound 对称整数量化教师模型 (W2 / W4 / W8, BF16) 校准与导出脚本。

功能说明:
    使用 WikiText-2 train 集的固定随机种子 (seed=42) token 窗口对 Qwen3.5-0.8B 进行
    AutoRound (SignRound / SignRoundV2) 校准量化，保护线性注意力中的 `.in_proj_a` 与
    `.in_proj_b` 层为 16-bit，其余线性层按指定位宽与 group_size 对称量化，并导出为
    `auto_round:auto_gptq` 格式及实验审计记录 `teacher_experiment.json`。

启动示例 (在 /workspace/my_project/Open-MOPD 目录下执行):
    1) 仅检查参数与算法配置 (不加载模型与 GPU):
       /usr/bin/python my_scripts/autoround_teacher.py \
           --bits 2 --group-size 32 --algorithm signroundv2 \
           --output /tmp/check_w2_cfg --check-config

    2) 校准生成高质量 W2 BF16 教师模型 (group_size=32, SignRoundV2, 512 条校准样本):
       HF_HOME=/cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       CUDA_VISIBLE_DEVICES=0 /usr/bin/python my_scripts/autoround_teacher.py \
           --model /workspace/my_project/models/Qwen3.5-0.8B \
           --bits 2 --group-size 32 --algorithm signroundv2 \
           --samples 512 --length 2048 --iters 200 \
           --output /data/models/Qwen3.5-0.8B-AutoRound-W2-BF16

    3) 校准生成 W8 BF16 教师模型 (group_size=128, SignRound, 128 条校准样本):
       HF_HOME=/cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       CUDA_VISIBLE_DEVICES=0 /usr/bin/python my_scripts/autoround_teacher.py \
           --model /workspace/my_project/models/Qwen3.5-0.8B \
           --bits 8 --group-size 128 --algorithm signround \
           --samples 128 --length 2048 --iters 200 \
           --output /data/models/Qwen3.5-0.8B-AutoRound-W8-BF16
"""

import argparse
import hashlib
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='使用 AutoRound 校准导出 W2/W4/W8 BF16 量化教师模型。')
    parser.add_argument('--model', default='/workspace/my_project/models/Qwen3.5-0.8B',
                        help='全精度基座模型路径')
    parser.add_argument('--output', required=True,
                        help='量化教师模型输出目录（必须尚不存在）')
    parser.add_argument('--samples', type=int, default=128,
                        help='从 WikiText-2 train 采样的校准窗口数量')
    parser.add_argument('--length', type=int, default=2048,
                        help='每条校准样本的序列长度')
    parser.add_argument('--iters', type=int, default=200,
                        help='每个量化块 SignRound 优化迭代步数')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='AutoRound 块优化时的微批次大小')
    parser.add_argument('--bits', type=int, choices=(2, 4, 8), default=8,
                        help='权重量化位宽（支持 2 / 4 / 8）')
    parser.add_argument('--group-size', type=int, choices=(32, 64, 128), default=128,
                        help='权重量化分组大小（W2 推荐 32，W4/W8 推荐 128）')
    parser.add_argument('--algorithm', choices=('signround', 'signroundv2'), default='signround',
                        help='量化舍入优化算法（极低位宽 W2 推荐 signroundv2）')
    parser.add_argument('--check-config', action='store_true',
                        help='仅打印解析后的 JSON 配置并退出，用于快速自检')
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise ValueError('Output must not already exist')
    if min(args.samples, args.length, args.iters, args.batch_size) < 1:
        parser.error('Calibration sizes and iterations must be positive')

    import torch
    from auto_round import AutoRound
    from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig, SignRoundV2Config
    from datasets import load_dataset
    from transformers import AutoProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration

    # 根据指定算法构造对应的 SignRound 配置对象
    algorithm_config = (SignRoundV2Config if args.algorithm == 'signroundv2' else SignRoundConfig)(
        iters=args.iters)
    if args.check_config:
        print(json.dumps(vars(args) | {'algorithm_config': type(algorithm_config).__name__}))
        return

    # 固定随机种子从 WikiText-2 train 全文中抽取不放回的固定长度校准窗口
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    processor = AutoProcessor.from_pretrained(args.model)
    dataset = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train')
    text = '\n\n'.join(dataset['text'])
    tokens = tokenizer(text, add_special_tokens=False)['input_ids']
    starts = random.Random(42).sample(range(len(tokens) - args.length + 1), args.samples)
    samples = [{'input_ids': torch.tensor(tokens[start:start + args.length]),
                'attention_mask': torch.ones(args.length, dtype=torch.long)} for start in starts]
    loader = torch.utils.data.DataLoader(samples, batch_size=1, shuffle=False)

    # 加载 BF16 基座模型，并将对量化敏感的线性注意力投影层 (in_proj_a / in_proj_b) 保留为 16-bit
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation='sdpa')
    preserved = {name: {'bits': 16} for name, module in model.named_modules()
                 if isinstance(module, torch.nn.Linear)
                 and (name.endswith('.in_proj_a') or name.endswith('.in_proj_b'))}

    output.mkdir(parents=True)
    record = vars(args) | dict(seed=42, split='train', dataset='Salesforce/wikitext/wikitext-2-raw-v1',
                              dataset_fingerprint=dataset._fingerprint,
                              text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                              token_window_starts=starts, dtype='bfloat16',
                              sym=True, preserved_layers=preserved,
                              algorithm_config=type(algorithm_config).__name__)
    (output / 'teacher_experiment.json').write_text(json.dumps(record, indent=2) + '\n')

    # 执行 AutoRound 校准并以 auto_round:auto_gptq 格式保存权重与分词器
    compressor = AutoRound(
        model, tokenizer, scheme={'bits': args.bits, 'act_bits': 16, 'group_size': args.group_size, 'sym': True,
                     'data_type': 'int'},
        alg_configs=algorithm_config,
        layer_config=preserved, dataset=loader, nsamples=args.samples, seqlen=args.length,
        iters=args.iters, batch_size=args.batch_size, device_map=0,
        enable_torch_compile=False, seed=42, processor=processor)
    compressor.quantize_and_save(str(output), format='auto_round:auto_gptq')
    tokenizer.save_pretrained(output)
    print(f'COMPLETE: {output}', flush=True)


if __name__ == '__main__':
    main()
