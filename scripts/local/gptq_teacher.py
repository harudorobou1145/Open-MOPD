import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--bits', type=int, choices=(4, 8), required=True)
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--length', type=int, default=2048)
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

    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train')
    tokens = tokenizer('\n\n'.join(dataset['text']), add_special_tokens=False)['input_ids']
    if len(tokens) < args.length:
        raise ValueError('Not enough calibration tokens')
    indices = random.Random(42).sample(range(len(tokens) - args.length + 1), args.samples)
    calibration = [{'input_ids': tokens[start:start + args.length],
                    'attention_mask': [1] * args.length} for start in indices]
    config = QuantizeConfig(bits=args.bits, group_size=128,
                            dynamic={r'-:.*(?:self_attn|linear_attn).*': {}}, device='cuda:0')
    model = GPTQModel.load(args.model, quantize_config=config,
                           torch_dtype=torch.float16, attn_implementation='sdpa')
    model.quantize(calibration, batch_size=1)
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