#!/usr/bin/env python3
"""GPTQ 仅量化 MLP 层 (W4 / W8, FP16) 教师模型构建脚本。

功能说明:
    针对早期仅对 Qwen3.5-0.8B 24 层 MLP 共 72 个线性层 (gate_proj / up_proj / down_proj)
    进行量化的实验设置，使用 GPTQModel 跳过所有注意力层 (`self_attn` 与 `linear_attn`)，
    仅将 72 个 MLP 投影层量化为 4-bit 或 8-bit (group_size=128, FP16)，并校验打包层数。

启动示例 (在 /workspace/my_project/Open-MOPD 目录下执行):
    1) 构建仅量化 MLP 的 W8 GPTQ 教师模型:
       HF_HOME=/cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       CUDA_VISIBLE_DEVICES=0 /usr/bin/python my_scripts/gptq_teacher.py \
           --model /workspace/my_project/models/Qwen3.5-0.8B \
           --bits 8 --samples 128 --length 2048 \
           --output /workspace/my_project/models/Qwen3.5-0.8B-GPTQ-MLP-W8-cal128-seq2048

    2) 构建仅量化 MLP 的 W4 GPTQ 教师模型:
       HF_HOME=/cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       CUDA_VISIBLE_DEVICES=0 /usr/bin/python my_scripts/gptq_teacher.py \
           --model /workspace/my_project/models/Qwen3.5-0.8B \
           --bits 4 --samples 128 --length 2048 \
           --output /workspace/my_project/models/Qwen3.5-0.8B-GPTQ-MLP-W4-cal128-seq2048
"""

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='使用 GPTQModel 构建仅量化 72 个 MLP 线性层的 W4/W8 教师。')
    parser.add_argument('--model', required=True, help='基座模型目录路径')
    parser.add_argument('--output', required=True, help='量化模型保存路径（必须尚不存在）')
    parser.add_argument('--bits', type=int, choices=(4, 8), required=True, help='MLP 量化位宽（4 或 8）')
    parser.add_argument('--samples', type=int, default=128, help='校准窗口样本数')
    parser.add_argument('--length', type=int, default=2048, help='每条校准窗口 token 长度')
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise ValueError('Output must not already exist')
    if args.samples <= 0 or args.length <= 0:
        raise ValueError('Samples and length must be positive')

    import torch
    from datasets import load_dataset
    from gptqmodel import GPTQModel, QuantizeConfig
    from transformers import AutoTokenizer

    # 从 WikiText-2 train 集采样固定种子随机窗口作为 GPTQ 校准集
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train')
    tokens = tokenizer('\n\n'.join(dataset['text']), add_special_tokens=False)['input_ids']
    if len(tokens) < args.length:
        raise ValueError('Not enough calibration tokens')
    indices = random.Random(42).sample(range(len(tokens) - args.length + 1), args.samples)
    calibration = [{'input_ids': tokens[start:start + args.length],
                    'attention_mask': [1] * args.length} for start in indices]

    # 通过负向正则跳过 self_attn 与 linear_attn，仅量化 MLP 线性层（与 AnyPrecision 学生量化范围对齐）
    config = QuantizeConfig(bits=args.bits, group_size=128,
                            dynamic={r'-:.*(?:self_attn|linear_attn).*': {}}, device='cuda:0')
    model = GPTQModel.load(args.model, quantize_config=config,
                           torch_dtype=torch.float16, attn_implementation='sdpa')
    model.quantize(calibration, batch_size=1)

    # 严格校验恰好有 24*3=72 个 MLP 投影层被量化打包
    packed = [name for name, module in model.model.named_modules() if hasattr(module, 'qweight')]
    if len(packed) != 72 or any('.mlp.' not in name for name in packed):
        raise ValueError(f'Expected 72 compressed MLP projections, got {packed}')

    model.save(str(output))
    tokenizer.save_pretrained(output)
    record = vars(args) | {'dataset': 'Salesforce/wikitext/wikitext-2-raw-v1', 'split': 'train',
                           'seed': 42, 'token_window_starts': indices,
                           'sampling': 'random windows from double-newline joined train text',
                           'dtype': 'float16', 'scope': 'MLP only',
                           'calibration_input_ids': [row['input_ids'] for row in calibration]}
    (output / 'teacher_experiment.json').write_text(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
