import argparse
import hashlib
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/workspace/my_project/models/Qwen3.5-0.8B')
    parser.add_argument('--output', required=True)
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--length', type=int, default=2048)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--bits', type=int, choices=(2, 4, 8), default=8)
    parser.add_argument('--group-size', type=int, choices=(32, 64, 128), default=128)
    parser.add_argument('--algorithm', choices=('signround', 'signroundv2'), default='signround')
    parser.add_argument('--check-config', action='store_true')
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

    algorithm_config = (SignRoundV2Config if args.algorithm == 'signroundv2' else SignRoundConfig)(
        iters=args.iters)
    if args.check_config:
        print(json.dumps(vars(args) | {'algorithm_config': type(algorithm_config).__name__}))
        return
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