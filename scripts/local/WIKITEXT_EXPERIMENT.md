# WikiText AnyPrecision MOPD

## W4-Only OPD Control

`run_w4_opd_bf16.sh` keeps the same parent8 cache, installed precisions 2-8,
and accepts `OPD_BITS=2` for the matched W2-only control. Both now default to
16 prompts, two sampled responses per prompt, minibatch 32, learning rate
1e-6 and 1485 updates, matching the dual-precision trajectory budget.
The default rollout is vLLM BF16; set `ANYPRECISION_ROLLOUT=hf` to use HF.
W2/BF16 uses the adapter's cached dense decoding, not a native INT2 kernel.
Final runs use GPUs 0/1 for W2 and 2/3 for W4, with output directories
`/data/anyprecision_mopd/parent8_w2_only_n2_vllm_full_20261113_final` and
`/data/anyprecision_mopd/parent8_w4_only_n2_vllm_full_20261113_final`.
Both passed two real updates with post-update vLLM synchronization. Step 2
took 18.28s for W2 and 18.78s for W4.

### Single-Precision Control Results

Both final runs completed all 1485 updates and exited 0. Excluding step 1,
W2 averaged 17.79s/step (generation 7.22s) and W4 17.58s/step (generation
7.36s), for 7.34h and 7.25h of summed step time. The earlier dual-precision
run averaged 16.30s/step and 7206 tokens/step; the new runs process 7264 to
7297 tokens/step, so the budgets are matched and throughput differs by ~8%.

WikiText-2 test PPL, identical non-overlapping-1024-block protocol, same
`before` state in every run:

| Model | W2 | W4 | W8 |
| --- | ---: | ---: | ---: |
| Initial student | 71.8543 | 20.5347 | 19.3854 |
| W2-only OPD | **43.4481** | 21.7072 | 21.0776 |
| W4-only OPD | 67.8946 | **20.0138** | 19.2325 |
| MOPD W2/W4 | 45.8258 | 20.1281 | 19.3447 |
| MOPD W2/W4/W8 | 46.7497 | 20.1813 | 19.4511 |
| Teachers | 27.9509 | 20.0120 | 19.4019 |

W2-only is the best W2 student (43.45, 5.2% below dual-precision MOPD) but
regresses W4 by 5.7% and W8 by 8.7%. W4-only reaches the W4 teacher (20.014
versus 20.012) and slightly improves W8, yet leaves W2 at 67.89. MOPD is the
only setting that improves every precision without regression.

Shared-codebook coupling explains the asymmetry. The 256-entry delta of every
trainable MLP is pooled by mean to build each precision's codebook, so one
update moves all precisions. Relative codebook shift vs the initial cache:

| Checkpoint | W2 | W4 | W8 |
| --- | ---: | ---: | ---: |
| W2-only | 0.0208 | 0.0163 | 0.0156 |
| W4-only | 0.0037 | 0.0056 | 0.0054 |

W2-only moved the W4/W8 codebooks almost as much as its own, and its raw
delta magnitude was 3x larger (1.97e-4 versus 6.6e-5) because the W2 gap to
its teacher was far larger. W4-only barely touched the W2 codebook. Treat this
as a mechanism hypothesis from two checkpoints, not a proven causal estimate.
Intermediate checkpoints were pruned (`max_actor_ckpt_to_keep=1`), so the
degradation timeline could not be measured.

vLLM explicitly loads W2/W4/W8 codebooks;
training-side precision checks and synchronization filters now include W2.
This matches the old dual-precision trajectory budget, not every old setting:
BF16, current parent8 cache and current microbatches remain unchanged.
On 2026-11-13 the earlier three-response HF controls were launched together:
W2-only on GPUs 0/1 under `/data/anyprecision_mopd/parent8_w2_only_n3_full_20261113_v2`,
and W4-only on GPUs 2/3 under `/data/anyprecision_mopd/parent8_w4_only_n3_full_20261113_v2`.
The first launches failed before training because actor validation omitted
`rollout.n` from the AnyPrecision batch size. This validation now uses the
expanded trajectory count; the original failed directories are retained.
The GPU pairs have different memory capacities (24 versus 48 GiB), so wall-time
differences are not a controlled strategy comparison. Check `train.log` and
`exit_status.txt` for completion; launch is not evidence of training success.

These earlier HF runs were stopped when switching to the dual-precision
budget. Their checkpoints must not be resumed for the new controls.

The controls retain the same
trainable shared codebook deltas, BF16 compute, optimizer and data as the
multi-precision presets. Only the selected precision generates trajectories
and receives gradients from its matching AutoRound teacher. Additional teachers are disabled. The single
precision uses the existing MT-OPD implementation with one teacher, not a
different full-parameter fine-tuning recipe. Default mode is dry-run.

```bash
bash scripts/local/run_w4_opd_bf16.sh --dry-run
OUTPUT_DIR=/data/anyprecision_mopd/parent8_w4_opd_control \
  bash scripts/local/run_w4_opd_bf16.sh --run
```

Evaluate W2/W4/W8 from the resulting checkpoint using the same test protocol
as previous runs. For a replicated checkpoint, pass one complete rank file:

```bash
HF_HOME=/cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=3 /usr/bin/python scripts/local/eval_anyprecision_wikitext.py \
  --cache '/data/anyprecision_w2w8/cache/quantized/(Qwen3.5-0.8B)-w8_orig2-gc1-c4_s100_blk512' \
  --checkpoint /data/anyprecision_mopd/parent8_w4_opd_control/checkpoints/global_step_1485/actor/model_world_size_2_rank_0.pt \
  --precisions 2 4 8 --model-precisions 2 3 4 5 6 7 8 \
  --output /data/anyprecision_mopd/parent8_w4_opd_control_ppl
```

Adjust step and world size to the actual saved checkpoint, and use a fresh
evaluation output directory. Compare pre-training and post-training PPL at all
three precisions, not only W4. Use held-out validation data for hyperparameter
and checkpoint selection, reserving test PPL for the final comparison.

With the default batch 16 and two responses, each single-precision control
consumes 32 trajectories per update, matching dual precision with one response
per precision. Set `N_RESPONSES=3` for the 48-trajectory triple-precision budget.
Prompt exposure and update counts match, but response lengths and
actual token budgets may differ. Report actual processed tokens and wall time.
Setting `N_RESPONSES=1` recovers the lower-compute single-response control.
A W4/W8-only control can further
separate the effect of adding W2. Do not infer superiority from training loss
or from a single final test score alone. Both v2 runs completed their first
real update: W2 loss 0.09024 / gradient norm 1.67898, W4 loss 0.01459 /
gradient norm 0.90908. These finite diagnostics validate startup, not final
training quality; completion and final W2/W4/W8 PPL remain pending.

## Replicated AnyPrecision Students

`run_w2w4_bf16.sh` and `run_w2w4w8_bf16.sh` default to GPUs 0/1/2/3
and FSDP1 `NO_SHARD`. Each GPU holds the complete parent8 student and its
optimizer states. Ranks process different trajectories and synchronize
gradients, without sharding or repeatedly gathering student parameters.
HF BF16 rollout, prompt batch 16, microbatch 4, and precision loss weights
are unchanged. Teacher placement is unchanged.

```bash
bash scripts/local/run_w2w4w8_bf16.sh --run
```

To select idle GPUs, set `CUDA_VISIBLE_DEVICES=2,3`, for example. To restore
the previous student sharding policy, append
`actor_rollout_ref.actor.fsdp_config.sharding_strategy=null`.
Other generic launcher defaults below are unaffected.

A real two-GPU W2/W4/W8 two-step test passed on 2026-11-13. Both ranks
saved complete model and optimizer states; all 1050 model tensors and 216
optimizer tensors matched exactly after updates. Step 2 took 52.05 seconds
(generation 36.61, actor update 9.11), with peak allocated 13.68 GiB and
reserved 19.84 GiB. Artifacts are under
`/data/anyprecision_mopd/parent8_w2w4w8_replicated_smoke_20261113_v3`.
GPU 0 was occupied, so four-GPU performance was not measured. Full-epoch
stability and resuming older sharded checkpoints in this mode are unverified;
use a fresh output directory rather than assuming checkpoint compatibility.

### Matched Two-GPU Performance Check

On 2026-11-13, FULL_SHARD and NO_SHARD each ran four steps sequentially on
GPUs 2/3, with identical W2/W4/W8 settings, prompt batch 16, microbatch 4,
and checkpoint saving disabled. Both exited successfully. Excluding step 1:

| Metric | FULL_SHARD | NO_SHARD |
| --- | ---: | ---: |
| Mean step time (seconds) | 70.71 | 62.58 |
| Mean generation time (seconds) | 38.01 | 37.31 |
| Mean actor update time (seconds) | 24.36 | 17.22 |
| Peak allocated memory over all four steps (GiB) | 12.48 | 13.67 |
| Peak reserved memory over all four steps (GiB) | 20.43 | 19.86 |

Step times for steps 2/3/4 were 81.05/76.21/54.88 seconds with FULL_SHARD
and 79.48/54.02/54.23 seconds with NO_SHARD. The measured mean speedup is
1.13x (11.5% less time), but both runs had large update-time spikes and
their fourth-step times were nearly identical. This short comparison does
not establish a stable speedup or isolate communication costs. Effective
token totals over steps 2-4 were 36256 and 36174 respectively; sampled
trajectories were not identical. GPU 0 remained occupied by another task.
Logs: `/data/anyprecision_mopd/perf_full_shard_20261113` and
`/data/anyprecision_mopd/perf_no_shard_20261113`.

Run from the repository root:

```bash
bash scripts/local/run_wikitext_experiment.sh --run
```

Defaults: GPUs 2 and 3, vLLM rollout, Qwen3.5-0.8B, WikiText, prompt 1024, response 128,
16 prompts per step, W4/W8 expansion to 32 trajectories, actor/rollout/student
scoring microbatch 8, teacher microbatch 2. Both precision groups contribute
gradients to one shared optimizer step. Two-GPU synchronous HF training is also
supported with the restrictions below.

## Four GPUs

Single-node synchronous training supports four GPUs with TP=1. Precision-dedicated
rollout assigns ranks 0/1 to W4 and ranks 2/3 to W8. Training remains shared FSDP;
teacher worker placement is unchanged. The global prompt batch must be divisible
by four. The launcher still defaults to GPUs 2/3 and does not enable multi-GPU tuning.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 TRAIN_BATCH_SIZE=16 \
  ACTOR_MICRO_BATCH_SIZE=4 LOGPROB_MICRO_BATCH_SIZE=4 \
  TEACHER_MICRO_BATCH_SIZE=1 \
  OUTPUT_DIR=/data/anyprecision_mopd/four_gpu_full \
  bash scripts/local/run_wikitext_experiment.sh --run
```

This is 16 prompts globally, expanded to 32 trajectories, not 16 prompts per GPU.
The W4:W8 loss weights remain 1:2. Use the same settings and output directory with
`trainer.resume_mode=auto` to resume a four-rank checkpoint. Do not directly resume
existing two-rank checkpoints on four ranks; world-size conversion is not implemented.

Verified on 2026-11-08 with two 24-GiB and two 48-GiB GPUs:

- Two short-context updates saved four model/optimizer/extra-state shards; a fresh
  launch restored model, optimizer, RNG and scheduler from step 2 and saved step 3.
- One standard-context update (prompt 1024, response limit 128) passed with finite
  gradient norm 0.38761, step time 17.01 seconds, reported peak allocated 12.83 GiB
  and reserved 14.73 GiB. These are logged metrics, not per-GPU external measurements.
- Artifacts: `/data/anyprecision_mopd/four_gpu_smoke_fixed_20261108` and
  `/data/anyprecision_mopd/four_gpu_standard_20261108`.

These checks do not establish full-epoch stability, convergence or a fixed speedup.
Generic routing tests cover W2/W4, but the launcher and real GPU checks still use
the existing W4/W8 cache and teachers; a shared W2/W4 student is not prepared here.

## Two GPUs

Precision-dedicated rollout is enabled by default for two GPUs
(rank 0 generates W4, rank 1 generates W8):

```bash
OUTPUT_DIR=/data/anyprecision_mopd/precision_split_probe \
  bash scripts/local/run_wikitext_experiment.sh --run \
  ++trainer.precision_per_gpu=True trainer.total_training_steps=2 trainer.save_freq=-1
```

This stably sorts generation tasks by precision before two-rank dispatch and
restores original sample order before teacher scoring and training. Both ranks
receive the same prompts, each at one precision. Training remains shared FSDP.
Requires one node, two GPUs and equal W4/W8 task counts. Single-GPU launches
disable it automatically. Override with `++trainer.precision_per_gpu=False`
to restore mixed-precision task distribution on two GPUs.
The standard-length two-step probe passed: generation 7.93/5.42 seconds,
whole step 22.89/18.87 seconds. Artifacts:
`/data/anyprecision_mopd/two_gpu_vllm_precision_split_fixed`.
These are sampled short-run results, not fixed-length benchmark or convergence evidence.

```bash
CUDA_VISIBLE_DEVICES=2,3 TRAIN_BATCH_SIZE=16 \
  OUTPUT_DIR=/data/anyprecision_mopd/two_gpu_full \
  bash scripts/local/run_wikitext_experiment.sh --run
```

The launcher sets `trainer.n_gpus_per_node=2`. Actor and student log-probability
microbatches default to 8. The global prompt batch must be even;
16 still means 16 prompts globally, expanded into 32 W4/W8 trajectories, not
16 prompts per GPU. Teacher settings and the shared student objective are unchanged.
Synchronous vLLM (TP=1 per GPU) and HF rollout are supported with FSDP training.
Batch tuning is not enabled for two GPUs. Precision grouping checks the number
of microbatches across ranks and falls back to unit microbatches if needed to
keep FSDP forward/backward collective counts aligned. vLLM receives complete
shared codebook tensors after each update, rather than FSDP shards. Pure-text
WikiText disables unused multi-modal model inputs. vLLM sleeps between rollout
phases and uses GPU memory utilization 0.3 to leave training headroom.

Standard-length vLLM probes (global batch 16, prompt 1024, response 128):
single GPU microbatch 8, step 2 = 27.66 seconds; two GPUs microbatch 8,
step 2 = 26.47 seconds (checkpoint saving separately took 6.31 seconds).
The earlier HF single-GPU step was 30.50 seconds. These short probes suggest
only a modest speed improvement, not a twofold gain; generation remains the
largest stage. Dual-GPU artifacts: `/data/anyprecision_mopd/two_gpu_vllm_mb8`.

A real two-GPU test with differing precision proportions verified rollout,
log-probabilities, finite updates and exact cross-rank equality of reconstructed
trainable parameters. A short-context MOPD run saved both rank shards at steps
1 and 2, then resumed model, optimizer, RNG and scheduler state to step 3.
Artifacts: `/data/anyprecision_mopd/two_gpu_smoke`.
A standard-length one-step run (global prompt batch 16, prompt 1024, response
128) also passed on physical GPUs 2 and 3: step time 136.10 seconds, reported
peak allocated 9.82 GiB and reserved 14.90 GiB, finite gradient norm 0.4994.
Artifacts: `/data/anyprecision_mopd/two_gpu_full_length_smoke`.
The earlier conservative HF unit-microbatch configuration is slower than the earlier
single-GPU large-microbatch configuration; it is not a throughput optimization.
These checks do not establish full-epoch stability, speedup, or convergence.
Checkpoints are FSDP world-size-2 shards; changing GPU count when resuming or
using the single-GPU PPL script requires checkpoint conversion first.

To resume the same two-GPU run, reuse its `OUTPUT_DIR` and append
`trainer.resume_mode=auto`. The default remains `trainer.resume_mode=disable`.

Outputs go to a timestamped directory under `/data/anyprecision_mopd`:
`train.log`, `tensorboard/`, `checkpoints/`, `batch_config.txt`,
`environment.txt`, `gpu_before.txt`, `launcher.pid`, and `exit_status.txt`.
The PID identifies the launcher, not every Ray child process.
Ray uses an isolated local instance with a short temporary path under `/data`.

```bash
# Print the command without running training.
bash scripts/local/run_wikitext_experiment.sh --dry-run

# Two steps, no checkpoint, console and TensorBoard logs retained.
OUTPUT_DIR=/data/anyprecision_mopd/my_smoke \
  bash scripts/local/run_wikitext_experiment.sh --run \
  trainer.total_training_steps=2 trainer.save_freq=-1

# Probe increasing microbatches; does not launch a full epoch afterward.
bash scripts/local/run_wikitext_experiment.sh --tune

# Explicit batch settings. Keep prompt batch divisible by actor microbatch.
TRAIN_BATCH_SIZE=32 ACTOR_MICRO_BATCH_SIZE=16 ROLLOUT_MICRO_BATCH_SIZE=16 \
  LOGPROB_MICRO_BATCH_SIZE=16 TEACHER_MICRO_BATCH_SIZE=2 CUDA_VISIBLE_DEVICES=3 \
  bash scripts/local/run_wikitext_experiment.sh --run

/usr/bin/python -m tensorboard.main \
  --logdir /data/anyprecision_mopd --port 6006
```

TensorBoard: http://localhost:6006 (forward port 6006 from the container).
Scalar tags include `actor/pg_loss`, `actor/grad_norm`, `timing_s/step`,
`perf/max_memory_allocated_gb`, and `perf/max_memory_reserved_gb`.

## Validation

On GPU 3 (48 GiB RTX 4090), real two-step original MOPD runs passed with
microbatches 4, 8, 16, and 20. Microbatch 24 ran out of memory. At microbatch
16, step 2 took 30.50 seconds, with PyTorch peak allocated 27.68 GiB and
reserved 37.88 GiB. Measurements exclude startup and do not establish
full-epoch stability or convergence. Defaults retain headroom for variable
sequences; repeat probing when changing prompt/response lengths or models.

The probe stops at the first CUDA OOM and records the largest passing value
in `best_microbatch.txt`. Other errors fail immediately. Each candidate uses
two training steps with checkpoint saving disabled. Failed runs retain logs.
Existing training jobs are not stopped by this launcher; select an idle GPU.