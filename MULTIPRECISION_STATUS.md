# Shared AnyPrecision Student In Original MOPD

## Storage and checkpoint follow-up (2026-11-04)

This follow-up supersedes the storage/checkpoint blocker recorded below.
After the host-side permission change, actual mkdir/touch/remove succeeded
under `/data/anyprecision_mopd` (about 17 TB available).

Executed the WikiText launcher with batch 1, 16 response tokens, one training
step and `trainer.save_freq=1`. The complete checkpoint was saved under
`/data/anyprecision_mopd/save_resume_check_20261104/checkpoints/global_step_1`:
model 3,944,924,555 bytes, optimizer 402,719,467 bytes, extra state,
data-loader state and model/tokenizer configuration.

A separate invocation used `trainer.resume_mode=resume_path`, that checkpoint
folder, `trainer.total_training_steps=2` and saving disabled. Logs confirm
model and optimizer loading followed by completion of step 2 with finite
gradient norm 2.4016196728 and no traceback. Logs are `save.log` and
`resume.log` in `/data/anyprecision_mopd/save_resume_check_20261104`.

Thus the short real-data generation/scoring/update/save/resume cycle is
verified with the W4 AWQ FP16 and local GPTQ W8 FP16 teachers. This does not
verify a full epoch, the default batch-8/response-128 configuration,
held-out PPL improvement, standalone packed export, or bitwise equivalence
between uninterrupted and resumed training. No full epoch was launched.

## WikiText + GPTQ W8 integration check (2026-11-04)

The local model is Qwen3.5-0.8B, not an 8B model. The independent W8
teacher is a GPTQ post-training quantization of the base, not a demonstrated
fine-tuned/distilled teacher. It has 72 packed MLP projections; attention
and other modules remain floating point. Do not use the failed W4-smoke
directory (zero quantized modules).

The original teacher worker failed to load this GPTQ checkpoint through
Transformers because that adapter requires the absent `optimum` package.
GPTQ checkpoints now use the already installed `GPTQModel.load` and return
its frozen underlying model. AWQ/compressed-tensors loading is unchanged.

Validated the original worker with the community W4 AWQ teacher (FP16,
`torch_awq`) and local `Qwen3.5-0.8B-GPTQ-MLP-W8-cal128-seq2048` teacher
(FP16, `torch`). Official tokenizer token IDs match both teachers. Sampled
token and top-k log probabilities are finite and agree for identical IDs.
GPU check log: `/tmp/gptq_teacher_worker_check.log`.

Ran the original Ray/FSDP trainer for two real WikiText steps using the
existing shared AnyPrecision cache, HF sequential W4/W8 rollout, batch 1
prompt expanded to two precision responses, top-k 4, 16-token responses,
1024-token prompt limit, seed 42 and learning rate 1e-6. Raw text template
was applied; no task verifier, validation, checkpoint save or resume was
performed. The train file was also supplied as the required val path, but
validation was disabled; it is NOT a held-out evaluation.

| Metric | Step 1 | Step 2 |
| --- | --- | --- |
| Pre-clipping gradient norm | 7.123154 | 2.401772 |
| Policy surrogate loss | 0.048938 | 0.007757 |
| Peak allocated GPU GiB | 10.912 | 11.294 |
| Step seconds | 10.181 | 7.139 |

Both precision microbatches reached backward each step and training completed
without a traceback. These are short plumbing checks, not convergence,
long-context memory measurements or evidence of quality improvement.
Training log: `/tmp/wikitext_gptq_mopd_check_20261104.log`.

Added `scripts/local/wikitext_anyprecision_mopd.sh`: defaults to the above
teacher pair with FP16 for BOTH, WikiText raw prompts, HF rollout, batch 8,
128 response tokens, one epoch and checkpoint frequency 100 (keep one actor
checkpoint). It uses `/data/anyprecision_mopd/wikitext_<timestamp>` and checks
output-parent write access before execution. Preview and execute from the repo:

```bash
bash scripts/local/wikitext_anyprecision_mopd.sh --dry-run
bash scripts/local/wikitext_anyprecision_mopd.sh --run
```

Use environment variables `OUTPUT_DIR`, `TRAIN_BATCH_SIZE`,
`MAX_RESPONSE_LENGTH`, `CUDA_VISIBLE_DEVICES` to change these defaults.
Trailing Hydra overrides can bound training, for example:

```bash
TRAIN_BATCH_SIZE=1 MAX_RESPONSE_LENGTH=16 \
bash scripts/local/wikitext_anyprecision_mopd.sh --run -- \
	trainer.total_training_steps=2 trainer.save_freq=1
```

This save-enabled command has NOT been executed: `/workspace` has only
about 2.1 GB free, and `/data` has about 17 TB free but is not writable by
this container session (nor by a check under its displayed owner UID).
The launcher correctly refuses to run with its current `/data` default.
The host/storage administrator must expose a writable project directory
and set `OUTPUT_DIR` accordingly. Put Ray spill/temp and future large caches
there as well once writable; do not delete existing artifacts automatically.
Checkpoint save/reload with this teacher pair remains unverified.

Before a quality claim or expensive full epoch: measure held-out WikiText
PPL for both student precisions and both teachers with identical scoring
windows, then compare pre/post training; prepare a separate validation set.
Batch 8 / 128-token default memory and throughput are not established by
the batch 1 / 16-token check. A full epoch has about 2970 optimizer updates
at batch 8 (depending on loader drop-last), not two steps. The earlier
GSM8K run does not establish WikiText improvement. A standalone packed
student export remains a separate unverified delivery requirement.

Regression: 10 precision-routing tests ran, 8 passed and 2 opt-in GPU tests
skipped; the teacher GPU test was run separately and passed with
`MOPD_GPU_TEST=1 MOPD_GPTQ_TEACHER=<absolute W8 GPTQ directory>`.

## WikiText MOPD input preparation (2026-11-02)

Added `scripts/local/prepare_wikitext.py`. Converted WikiText-2 raw train
to `/workspace/my_project/data/wikitext2_mopd/train.parquet`, with a sibling
`manifest.json`. Of 36718 source rows, 23767 nonblank rows were retained.
Text, titles and source order are preserved; no chunking, shuffle, answers
or validation/test rows were added. All retained rows fit 1024 tokens with
the local Qwen3.5 tokenizer. Short rows and titles remain in the training pool.

Use this file as the launcher's training input, keep true task reward disabled,
and explicitly set the following data configuration (not the default Qwen
chat template):

```yaml
data:
	apply_chat_template_kwargs:
		chat_template: "{{ messages[0]['content'] }}"
```

Each stored prompt is one message for schema compatibility, but this template
emits only its raw content, without user/assistant markers. The student must
generate its own continuation; the source text is not a supervised target.
No teacher precision/domain label is encoded in the data: existing precision
pairing supplies it during training. No held-out validation input was prepared.

Validation: conversion checked template equality for every retained row.
The original RLHFDataset with tokenizer-only processing loaded all 23767 rows,
and first/last raw_prompt_ids matched the original text tokenization exactly.
Processor-based integration and an actual training step were not run.
This preparation did not launch training or PPL evaluation or change launch
defaults. Data files were not committed or uploaded.

## W8 GPTQ calibration run (2026-11-02)

Rebuilt the W8 FP16 candidate from the original Qwen3.5-0.8B with
128 calibration windows of 2048 tokens (262144 sampled tokens, windows may
overlap). The script now samples unique window starts with seed 42 from
double-newline joined WikiText-2 raw train text; validation/test are not used.
Defaults changed from the 16x256 probe to 128x2048. Group size remains 128,
MLP-only scope remains unchanged, and attention is not quantized.

Output: `/workspace/my_project/models/Qwen3.5-0.8B-GPTQ-MLP-W8-cal128-seq2048`.
Calibration arguments, window starts and token IDs are saved in
`teacher_experiment.json`. The previous probe outputs were not overwritten.
Logs: `/workspace/my_project/outputs/gptq_w8_cal128_seq2048.log` and
`/workspace/my_project/outputs/gptq_w8_cal128_seq2048_verify.log`.

Validation: script syntax and window-bound checks passed. The saved model
reloaded using GPTQModel's torch backend, FP16 and SDPA; all 72 packed MLP
modules observed FP16 inputs on the smoke passage. Loss was 2.947725772857666
and logits were finite. This is not full WikiText PPL, evidence of superiority
over the student, or verification of MOPD teacher integration. Internal FP32
operations and generic backend weight reconstruction may still occur.
No MOPD training or model upload was performed. Disk remaining: about 4.2 GB.

## GPTQ teacher probe (2026-11-02)

Added `scripts/local/gptq_teacher.py` to build independent MLP-only W4/W8
teachers from the same local Qwen3.5-0.8B base using GPTQModel 5.8.0.
Calibration uses Salesforce/wikitext, wikitext-2-raw-v1 train only, seed 42,
16 shuffled eligible text rows truncated to 256 tokens, group size 128 and
FP16 loading. This is a compatibility probe, not final calibration quality.

The old dataset alias failed Hub URI parsing; the full namespace works.
An initial negative-lookahead dynamic exclusion produced zero quantized
modules. The script now excludes attention explicitly and requires exactly
72 packed MLP projections before saving. The failed output remains at
`/workspace/my_project/models/Qwen3.5-0.8B-GPTQ-MLP-W4-smoke` and must NOT
be used as a quantized teacher.

Successful local outputs:

- `/workspace/my_project/models/Qwen3.5-0.8B-GPTQ-MLP-W4-probe2`
- `/workspace/my_project/models/Qwen3.5-0.8B-GPTQ-MLP-W8-smoke`

Validation: both saved models reload with backend `torch`, each has 72
qweight modules, a sampled quantized module receives FP16 input, and the
same short text produces finite next-token loss. Saved calibration token
IDs match exactly across bit widths. The loader produces a text-only model;
integration with original MOPD teacher loading is not yet tested. Generic
torch quantized execution may reconstruct weights and is not a verified
fused packed-kernel teacher runtime. Internal operations may use FP32;
the FP16 observation does not assert every intermediate has that dtype.

W8 FP16 recheck (2026-11-02): reused the existing W8 output without another
quantization run or model copy. Loaded with `backend=torch`, FP16 and SDPA;
forward pre-hooks on all 72 packed MLP modules observed only FP16 inputs
on one short English passage. Next-token loss was 2.9445641040802 and all
logits were finite. This is a smoke check, not WikiText PPL or proof that
all internal accumulations use FP16. Reference fallbacks for causal_conv1d
and gated delta rule were reported; throughput is not established.

No full WikiText validation/test PPL, teacher-versus-student comparison,
MOPD training, Git commit or GitHub upload was performed for this probe.
Remaining disk space after the experiment is approximately 6.2 GB; avoid
creating more model copies before reviewing storage. Each output contains
`teacher_experiment.json` with arguments and actual calibration token IDs.

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