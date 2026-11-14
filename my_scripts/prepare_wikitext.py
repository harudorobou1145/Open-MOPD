#!/usr/bin/env python3
"""WikiText-2 训练数据预处理脚本 (生成 Open-MOPD Parquet 格式)。

功能说明:
    读取 HuggingFace 上的 `Salesforce/wikitext` (`wikitext-2-raw-v1`, `train` 划分)，
    仅剔除纯空白行，原样保留段落文本、顺序与标题，封装为 Open-MOPD 所需的 `prompt` 消息列，
    并校验纯文本续写模板 `{{ messages[0]['content'] }}` 不会注入多余特殊字符或对话前缀，
    最终输出 `train.parquet` 与数据统计清单 `manifest.json`。

启动示例 (在 /workspace/my_project/Open-MOPD 目录下执行):
    1) 生成默认最大提示长度 1024 的 WikiText-2 MOPD 训练集:
       HF_HOME=/cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       /usr/bin/python my_scripts/prepare_wikitext.py \
           --model /workspace/my_project/models/Qwen3.5-0.8B \
           --output /workspace/my_project/data/wikitext2_mopd

    2) 自定义最大提示词过滤长度统计阈值:
       /usr/bin/python my_scripts/prepare_wikitext.py \
           --model /workspace/my_project/models/Qwen3.5-0.8B \
           --max-prompt-length 512 \
           --output /tmp/wikitext2_mopd_len512
"""

import argparse
import json
from pathlib import Path


# 原样透传用户文本内容的 Jinja2 模板，避免引入 <|im_start|> 等聊天角色标记
RAW_TEMPLATE = "{{ messages[0]['content'] }}"


def main():
    parser = argparse.ArgumentParser(description='将 WikiText-2 train 集转换为 MOPD Parquet 格式。')
    parser.add_argument('--model', required=True, help='用于统计 token 长度与验证模板的分词器模型目录')
    parser.add_argument('--output', required=True, help='输出的全新目录路径（自动生成 train.parquet 和 manifest.json）')
    parser.add_argument('--max-prompt-length', type=int, default=1024,
                        help='统计满足长度上限的样本数（实际过滤在训练 DataLoader 中按此长度执行）')
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise ValueError('Output must not already exist')

    from datasets import Dataset, load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    source = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train')

    rows = []
    eligible = 0
    for index, item in enumerate(source):
        text = item['text']
        # 仅跳过空行，保留所有正文与章节标题行
        if not text.strip():
            continue
        messages = [{'role': 'user', 'content': text}]
        # 验证自定义模板渲染后与原始文本逐字符一致
        rendered = tokenizer.apply_chat_template(
            messages, chat_template=RAW_TEMPLATE, tokenize=False,
            add_generation_prompt=True,
        )
        if rendered != text:
            raise ValueError('Raw continuation template changed the source text')
        length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
        eligible += int(length <= args.max_prompt_length)
        # 构造 verl / Open-MOPD 标准数据字段；纯蒸馏任务不使用 ground_truth 规则打分
        rows.append({'prompt': messages, 'data_source': 'wikitext2',
                     'ability': 'text_continuation',
                     'reward_model': {'style': 'rule', 'ground_truth': ''},
                     'extra_info': {'index': index, 'split': 'train',
                                    'prompt_tokens': length}})
    if not rows:
        raise ValueError('No nonempty training rows')

    output.mkdir(parents=True)
    Dataset.from_list(rows).to_parquet(str(output / 'train.parquet'))
    manifest = {'dataset': 'Salesforce/wikitext', 'config': 'wikitext-2-raw-v1',
                'split': 'train', 'source_rows': len(source), 'saved_rows': len(rows),
                'model': args.model, 'max_prompt_length': args.max_prompt_length,
                'eligible_rows': eligible, 'chat_template': RAW_TEMPLATE,
                'processing': 'Drop blank rows only; preserve text, order and titles.',
                'notes': 'No labels, chunking, shuffle or held-out data used. Disable true task reward.'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
