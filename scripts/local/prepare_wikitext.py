import argparse
import json
from pathlib import Path


RAW_TEMPLATE = "{{ messages[0]['content'] }}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-prompt-length', type=int, default=1024)
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
        if not text.strip():
            continue
        messages = [{'role': 'user', 'content': text}]
        rendered = tokenizer.apply_chat_template(
            messages, chat_template=RAW_TEMPLATE, tokenize=False,
            add_generation_prompt=True,
        )
        if rendered != text:
            raise ValueError('Raw continuation template changed the source text')
        length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
        eligible += int(length <= args.max_prompt_length)
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