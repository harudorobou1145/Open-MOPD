# Shared AnyPrecision Student In Original MOPD

## Experiment update (2026-11-01)

This update supersedes the historical HF-only limits below.

- Added precision-paired data expansion, teacher routing, actor precision
	contexts and shared-codebook installation/configuration without replacing
	the original MOPD objective or optimizer loop.
- Added native compressed vLLM rollout selection, sequential W4/W8 generation
	and actor-codebook synchronization. HF remains the default launcher backend.
- Added quantized teacher loading compatibility and Arrow chunk handling.
- Added graceful time-budget checkpoint/exit, the budget/evaluation launcher
	`scripts/local/anyprecision_budget.py` and precision-routing regression tests.
- Completed 26 original Ray/FSDP updates on one 48 GB GPU: batch 8 prompts,
	16 W4/W8 responses, actor microbatch 1, learning rate 1e-6, prompt/response
	limits 1024/2048. Final checkpoint reloaded for compressed evaluation.
- Mean step 203.85 s: generation 143.27 s, actor update 40.54 s, reward 20.00 s.
	Mean response length 1225.82 tokens; mean length-limit fraction 54.6%.
- Logged policy loss averaged 0.003913 over the first 8 steps and 0.003514 over
	the last 8; this changing-batch on-policy surrogate is not a fixed-data
	convergence metric. Pre-clipping gradient norm ranged 0.286 to 6.859,
	with no nonfinite values; the clipping threshold was 1.0.
- Fixed 256 GSM8K test questions, seed 42, greedy, thinking disabled:
	strict answer scoring W4 32 -> 35, W8 36 -> 36 correct; relaxed numeric
	scoring W4 127 -> 131, W8 145 -> 144. Both relaxed paired bootstrap 95%
	difference intervals include zero. No reliable joint improvement is shown.

Remaining limitations: actor expands floating MLP weights and owns storage
separately from rollout; compressed prefill uses small-row kernel calls;
no controlled throughput comparison or standalone compressed export exists.
W4 teacher uses FP16 while W8 uses decompressed BF16; this compute-dtype
confound has not been isolated. Fixed-trajectory convergence metrics and
per-precision gradient diagnostics are not logged. Model/data/checkpoint
artifacts are intentionally excluded from Git.

Publication validation: `python -m unittest discover -s training/verl/tests
-p test_precision_routing.py` ran 10 tests, 8 passed and 2 optional GPU tests
skipped. `git diff --check` passed. This publication did not rerun training.

## Scope

Initially validated on 2026-10-31 using container-default `/usr/bin/python`, not conda.
The 2026-11-01 experiment update below supersedes earlier HF-only scope limits.
The original `verl.trainer.main_ppo`, MT-OPD reward, advantage calculation and
actor optimizer are retained. The rejected simplified OPD runner remains removed.

The Qwen3.5-0.8B student uses the sibling `any_precision.trainable` module:
72 MLP projections share trainable parent-codebook deltas, with fixed prefix
indices and initial codebooks. This is MLP-only quantization, not whole-model
quantization or verified packed-kernel training.

Each input is expanded into W4 and W8 student rollouts before generation. Both
use the same actor parameters; precision metadata selects the actual codebooks
for generation, scoring and backward. Matching teachers are selected with
`mt_opd.domain_weighting=precision_routing` and `teacher_bits=[4,8]`.
All precision microbatches accumulate before one original optimizer step.

## Verified

- Original frozen reward-worker sampled-token and top-k scoring for both teachers.
- Official tokenizer vocabulary IDs match both teachers for all official tokens.
	Student token IDs are scored directly without independent teacher retokenization.
- W4 AWQ uses GPTQModel 5.8.0 `torch_awq`, FP16 and SDPA.
- W8 uses compressed-tensors and BF16. Its loader decompresses weights; this is
	not evidence of a packed INT8 runtime.
- Original HF rollout directly shares the FSDP actor. Two-dimensional text and
	batch-first four-channel position tensors are covered by GPU tests.
- Original trainer completed step 1 with both teachers and finite gradient norm
	15.797561645507812; model, optimizer and extra state were saved.
- Original trainer loaded that checkpoint, optimizer, RNG and scheduler,
	evaluated W8 and continued to step 2 with finite gradient norm 39.08397674560547.
- Eight focused tests passed, including the two optional real-GPU tests.

These are four-token, top-k=4 plumbing checks on two identical arithmetic prompts,
not evidence that distillation improves task accuracy. W8 validation scored zero.
Validation defaults to the highest configured precision; set
`+actor_rollout_ref.model.anyprecision.eval_bits=4` for a separate W4 evaluation.
It does not automatically report both precisions.

## Run

Use the original launcher with W4 first and W8 second. Set `ANYPRECISION_CACHE`
to the existing quantized cache directory containing `weights` and `lut_*`, and
`ANYPRECISION_REPO` to the sibling repository if it is not at the default location.

```bash
export CUDA_VISIBLE_DEVICES=2
export ANYPRECISION_CACHE='/path/to/quantized/(Qwen3.5-0.8B)-w8_orig4-gc1-c4_s100_blk512'
PYTHON_BIN=/usr/bin/python MAX_PROMPT_LENGTH=64 MAX_RESPONSE_LENGTH=4 \
bash scripts/local/mt_opd.sh --run \
	--model /workspace/my_project/models/Qwen3.5-0.8B \
	--teacher /workspace/my_project/models/Qwen3.5-0.8B-W4A16-AutoRound-AWQ \
	--teacher /workspace/my_project/models/Qwen3.5-0.8B-W8A16 \
	--train /workspace/my_project/outputs/anyprecision_original_mopd_smoke/smoke.parquet \
	--val /workspace/my_project/outputs/anyprecision_original_mopd_smoke/smoke.parquet \
	--output /workspace/my_project/outputs/anyprecision_original_mopd_smoke/run -- \
	trainer.total_training_steps=1 trainer.val_before_train=False \
	trainer.test_freq=-1 trainer.save_freq=-1 data.dataloader_num_workers=0 \
	++actor_rollout_ref.rollout.log_prob_top_k=4 \
	actor_rollout_ref.actor.use_kl_loss=False algorithm.use_kl_in_reward=False
```

Omit `--run` to preview. Use a new output directory for a fresh experiment,
or explicitly configure `trainer.resume_mode=resume_path` and
`trainer.resume_from_path` to restore an existing global-step directory.
Keep `data.dataloader_num_workers` unchanged when restoring a checkpoint;
the original stateful loader cannot restore multiprocess state with zero workers.
Teacher loading requires the tested GPTQModel auxiliary packages and
`qwen-vl-utils==0.0.14`; core torch/Transformers versions were not upgraded.

## Limits

Currently constrained to single-GPU synchronous HF rollout, FSDP with original
parameters, FP32 model initialization, padded/unfused inputs, size-one actor,
rollout and log-probability microbatches, and one PPO epoch/minibatch per paired
batch. Multi-GPU, vLLM student rollout, multimodal prompts and packed export are
not validated. Keep cross-teacher conflict replacement disabled for this baseline.

Full original FSDP checkpoints include frozen weights and buffers, about 4.1 GB
for the first smoke checkpoint. Disk filled during a second save; only that
incomplete step-2 directory was removed. The complete step-1 checkpoint remains.
The resumed test disabled another save. Earlier runs also emitted DataLoader
worker cleanup warnings after training; these are distinct from the completed
training update. Free disk space before running a longer experiment.

Logs: `/workspace/my_project/outputs/anyprecision_original_mopd_smoke/trainer.log`
and `resume.log`. No commit or push was performed.

## Tests

```bash
CUDA_VISIBLE_DEVICES=2 MOPD_GPU_TEST=1 \
PYTHONPATH="$PWD/training/verl:$PWD/../any-precision-llm" \
/usr/bin/python -m unittest discover -s training/verl/tests -p test_precision_routing.py
```
Tests cover teacher ordering, invalid mappings, original domain routing,
metadata preservation in both generation modes and the opt-in teacher loader
arguments and guards. They do not require Ray or vLLM.
No commit or push has been performed.