import argparse
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time


def evaluate(args):
    import pandas as pd
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from any_precision.integrations.vllm.control import set_precision, update_from_actor

    rows = pd.read_parquet(args.data)
    if not 1 <= args.count <= len(rows):
        raise ValueError(f'count must be between 1 and {len(rows)}')
    indices = random.Random(42).sample(range(len(rows)), args.count)
    tokenizer = AutoTokenizer.from_pretrained(os.environ['MODEL_PATH'])
    prompts = [tokenizer.apply_chat_template(
        [{'role': 'user', 'content': rows.iloc[index]['question'] + '\nPut the final numerical answer after ####.'}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False) for index in indices]
    engine = LLM(model=os.environ['MODEL_PATH'], quantization='anyprecision', load_format='anyprecision',
                 model_loader_extra_config={'cache_path': os.environ['ANYPRECISION_CACHE']}, dtype='float16',
                 tensor_parallel_size=1, enforce_eager=True, max_model_len=2048,
                 max_num_seqs=8, max_num_batched_tokens=2048, gpu_memory_utilization=0.5,
                 enable_prefix_caching=False, limit_mm_per_prompt={'image': 0, 'video': 0})
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        state = {key: value for key, value in state.items()
                 if key.endswith(('.delta', '.base_lut4', '.base_lut8'))}
        if not state or not any(torch.count_nonzero(value).item() for key, value in state.items()
                                if key.endswith('.delta')):
            raise ValueError('Checkpoint has no updated shared codebooks')
        engine.apply_model(lambda model: update_from_actor(model, state.items()))
    result = {'dataset': 'openai/gsm8k/main/test', 'seed': 42, 'count': len(indices),
              'checkpoint': args.checkpoint, 'max_tokens': 1024, 'precisions': {}}
    for bits in (4, 8):
        engine.reset_prefix_cache()
        engine.apply_model(lambda model: set_precision(model, bits))
        outputs = engine.generate(prompts, SamplingParams(temperature=0, max_tokens=1024), use_tqdm=True)
        records = []
        for index, output in zip(indices, outputs, strict=True):
            text = output.outputs[0].text
            expected = rows.iloc[index]['answer'].split('####')[-1].strip().replace(',', '')
            matches = re.findall(r'####\s*(-?[\d,]+(?:\.\d+)?)', text)
            predicted = matches[-1].replace(',', '') if matches else None
            records.append({'index': index, 'expected': expected, 'predicted': predicted,
                            'correct': predicted == expected, 'output': text,
                            'tokens': len(output.outputs[0].token_ids),
                            'finish_reason': output.outputs[0].finish_reason})
        result['precisions'][str(bits)] = {'accuracy': sum(row['correct'] for row in records) / len(records),
                                           'records': records}
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f'W{bits}: {sum(row["correct"] for row in records)}/{len(records)} correct', flush=True)


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve()
    environment = dict(os.environ)
    root = script.parents[2]
    environment.update(CUDA_VISIBLE_DEVICES='2', PYTHONUNBUFFERED='1',
                       VLLM_ENABLE_V1_MULTIPROCESSING='0', ANYPRECISION_ROLLOUT='vllm',
                       TRAIN_BATCH_SIZE='8', PYTHON_BIN=sys.executable,
                       TRAIN_FILE='/workspace/my_project/data/Open-MOPD-Data/rl_prompt_mix/train.parquet',
                       VAL_FILE='/workspace/my_project/data/Open-MOPD-Data/rl_prompt_mix/train.parquet',
                       OUTPUT_DIR=str(output), CHECKPOINT_DIR=str(output / 'checkpoints'),
                       MODEL_PATH='/workspace/my_project/models/Qwen3.5-0.8B',
                       ANYPRECISION_CACHE='/workspace/my_project/outputs/anyprecision_qwen35_08b/'
                       'mlp_seed4_parent8_s0/calibration_cache/quantized/(Qwen3.5-0.8B)-w8_orig4-gc1-c4_s100_blk512',
                       TEACHER_MODEL_PATHS='/workspace/my_project/models/Qwen3.5-0.8B-W4A16-AutoRound-AWQ,'
                       '/workspace/my_project/models/Qwen3.5-0.8B-W8A16',
                       PYTHONPATH=f'{root}/training/verl:{root.parent}/any-precision-llm')
    started = time.time()

    def execute(command, name):
        with (output / f'{name}.log').open('w') as stream:
            subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)

    status = {'started': started, 'target_seconds': 7200, 'status': 'baseline'}
    status_path = output / 'status.json'
    status_path.write_text(json.dumps(status, indent=2))
    try:
        evaluation = [sys.executable, str(script), 'evaluate', '--data', args.data, '--count', str(args.count)]
        execute(evaluation + ['--output', str(output / 'baseline.json')], 'baseline')
        budget = min(5400, max(60, int(7200 - (time.time() - started) - 900)))
        status.update(status='training', training_budget_seconds=budget)
        status_path.write_text(json.dumps(status, indent=2))
        execute(['bash', str(script.parent / 'mt_opd.sh'), '--run', '--',
                 'trainer.total_training_steps=null', 'trainer.val_before_train=False', 'trainer.test_freq=-1',
                 'trainer.save_freq=5', 'trainer.max_actor_ckpt_to_keep=1', 'trainer.resume_mode=disable',
                 f'+trainer.time_budget_seconds={budget}', 'data.dataloader_num_workers=0', 'data.seed=42',
                 '++actor_rollout_ref.rollout.log_prob_top_k=4', 'actor_rollout_ref.actor.use_kl_loss=False',
                 'algorithm.use_kl_in_reward=False'], 'train')
        checkpoints = list((output / 'checkpoints').glob('global_step_*/actor/model_world_size_1_rank_0.pt'))
        checkpoint = max(checkpoints, key=lambda path: int(path.parents[1].name.split('_')[-1]))
        status.update(status='final_evaluation', checkpoint=str(checkpoint))
        status_path.write_text(json.dumps(status, indent=2))
        execute(evaluation + ['--output', str(output / 'final.json'), '--checkpoint', str(checkpoint)], 'final')
        status.update(status='completed', elapsed_seconds=time.time() - started)
    except Exception as error:
        status.update(status='failed', error=str(error), elapsed_seconds=time.time() - started)
        raise
    finally:
        status_path.write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['run', 'evaluate'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--count', type=int, default=16)
    arguments = parser.parse_args()
    (run if arguments.mode == 'run' else evaluate)(arguments)