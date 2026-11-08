# WikiText AnyPrecision MOPD

Run from the repository root:

```bash
bash scripts/local/run_wikitext_experiment.sh --run
```

Defaults: GPUs 2 and 3, vLLM rollout, Qwen3.5-0.8B, WikiText, prompt 1024, response 128,
16 prompts per step, W4/W8 expansion to 32 trajectories, actor/rollout/student
scoring microbatch 8, teacher microbatch 2. Both precision groups contribute
gradients to one shared optimizer step. Two-GPU synchronous HF training is also
supported with the restrictions below.

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