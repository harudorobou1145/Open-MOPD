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
    path = Path(checkpoint)
    if not path.is_dir():
        return torch.load(path, map_location='cpu', weights_only=False, mmap=True)
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
    return restored if rank == 0 else None


def main():
    parser = argparse.ArgumentParser(description='Compare pre/post MOPD WikiText-2 test perplexity.')
    parser.add_argument('--model', default='/workspace/my_project/models/Qwen3.5-0.8B')
    parser.add_argument('--anyprecision-repo', default='/workspace/my_project/any-precision-llm')
    parser.add_argument('--cache', default=(
        '/workspace/my_project/outputs/anyprecision_qwen35_08b/mlp_seed4_parent8_s0/'
        'calibration_cache/quantized/(Qwen3.5-0.8B)-w8_orig4-gc1-c4_s100_blk512'))
    parser.add_argument('--checkpoint', default=(
        '/data/anyprecision_mopd/batched_full_20261105_085300/checkpoints/'
        'global_step_1485/actor/model_world_size_1_rank_0.pt'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--sequence-length', type=int, default=1024)
    parser.add_argument('--max-blocks', type=int)
    parser.add_argument('--standalone-model')
    parser.add_argument('--quant-backend')
    parser.add_argument('--include-teachers', action='store_true')
    parser.add_argument('--teacher-w4', default='/workspace/my_project/models/Qwen3.5-0.8B-W4A16-AutoRound-AWQ')
    parser.add_argument('--teacher-w8', default='/workspace/my_project/models/Qwen3.5-0.8B-GPTQ-MLP-W8-cal128-seq2048')
    args = parser.parse_args()
    if args.sequence_length < 2 or (args.max_blocks is not None and args.max_blocks < 1):
        parser.error('sequence-length >= 2 and max-blocks >= 1 required')
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
    manifest = dict(dataset='Salesforce/wikitext', config='wikitext-2-raw-v1', split='test',
                    dataset_fingerprint=dataset._fingerprint,
                    text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    source_tokens=tokens.numel(), scored_tokens=sum(b.numel() - 1 for b in blocks),
                    blocks=len(blocks), sequence_length=args.sequence_length,
                    protocol='Non-overlapping blocks, include final partial block, omit each block first token; '
                             'no chat template or added special tokens; summed FP32 token cross entropy.',
                    dtype='FP32 model and delta with BF16 autocast', model=args.model,
                    cache=args.cache, checkpoint=args.checkpoint,
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    smoke_test=args.max_blocks is not None,
                    standalone_model=args.standalone_model,
                    teachers=({'4': args.teacher_w4, '8': args.teacher_w8} if args.include_teachers else {}),
                    teacher_dtype='FP16, matching training; W4 torch_awq, W8 torch',
                    independence_note='Official test split; calibration contamination has not been audited.')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
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
        install_trainable_qwen(model, args.cache, precisions=(4, 8))
    model.eval().requires_grad_(False)
    model.to('cuda')
    results = []
    stages = ('before', 'after', 'teacher') if args.include_teachers else ('before', 'after')
    if args.standalone_model:
        stages = ('standalone',)
    for stage in stages:
        if stage == 'after':
            state = checkpoint_state if checkpoint_state is not None else load_checkpoint(args.checkpoint)
            model.load_state_dict(state, strict=True)
            checkpoint_state = None
            del state
        for bits in ((None,) if args.standalone_model else (4, 8)):
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
            precision_context = nullcontext() if stage in ('teacher', 'standalone') else student_precision(model, bits)
            with torch.inference_mode(), precision_context:
                for index, block in enumerate(blocks, 1):
                    inputs = block.to('cuda')
                    with torch.autocast('cuda', dtype=torch.float16 if stage == 'teacher' else torch.bfloat16):
                        logits = model(input_ids=inputs, use_cache=False).logits
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