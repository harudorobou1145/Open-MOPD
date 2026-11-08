# Teacher Candidate PPL Results

Completed 2026-11-06. Lower PPL is better.

## Protocol

WikiText-2 raw official test, text joined with two newlines, reference Qwen tokenizer,
no special tokens or chat template. Non-overlapping 1024-token blocks including
the final partial block; omit each block's first token. 291 blocks, 296902 scored
tokens. Sum FP32 cross entropy before exponentiating. Candidate vocabularies
were checked against the reference vocabulary.

## Results

| Model | Execution | PPL |
| --- | --- | ---: |
| Original Qwen3.5-0.8B | BF16 | 19.389505 |
| Intel int4 AutoRound candidate | BF16, auto_round:torch_zp | 20.011975 |
| axteam AWQ-8bit candidate | BF16, compressed-tensors decompressed weights | 19.415161 |
| Local AutoRound W8 teacher | BF16, auto_round:torch_zp | 19.401943 |
| Previous W4 AutoRound AWQ teacher | FP16, torch_awq | 20.179023 |
| Previous W8 GPTQ MLP teacher | FP16, torch | 19.378357 |
| AnyPrecision W4 before MOPD | FP32 parameters, BF16 autocast | 20.581855 |
| AnyPrecision W4 after MOPD | FP32 parameters, BF16 autocast | 19.806865 |
| AnyPrecision W8 before MOPD | FP32 parameters, BF16 autocast | 19.370385 |
| AnyPrecision W8 after MOPD | FP32 parameters, BF16 autocast | 19.403236 |

New W4 improves over the previous W4 teacher, but does not beat the trained W4
student. New W8 does not improve on the previous W8 teacher in this evaluation.
Different execution dtypes and quantization scopes mean this is an end-to-end
comparison, not an isolated quantization-algorithm experiment. Tiny differences
around the BF16 baseline do not establish statistical superiority. Teacher PPL
alone does not establish distillation quality. Use validation data for further
selection; calibration/test contamination has not been audited.

## Artifacts

- Intel revision: `5027a5784f16e6a4f3fe50b9aba89d1ad4dea1d0`
- Intel snapshot: `/data/models/Qwen3.5-0.8B-int4-AutoRound`
- axteam revision: `92064fa5cc0b2fe124dc8a413e8635ba49847cb7`
- axteam snapshot: `/data/models/Qwen3.5-0.8B-AWQ-8bit`
- Download endpoint: `https://hf-mirror.com`, fixed revisions, all listed repository files.
- New results: `/data/anyprecision_mopd/new_teachers/{w4,w8,bf16}_final/`
- Each result directory contains `manifest.json` and `results.json`; sibling `.log` files contain full output.
- Previous six-group results: `/data/anyprecision_mopd/eval_with_teachers_20261106/`
- Installed `auto-round==0.16.1` with `--no-deps`; existing Torch/Transformers unchanged.
- W4 automatic Marlin selection failed because its CUDA extension is absent;
  the explicitly selected Torch zero-point backend passed smoke and full evaluations.
- W8 Transformers loading decompresses quantized weights. These runs do not
  benchmark packed INT8/INT4 kernels or establish packed serving compatibility.

## Reproduce

### Local AutoRound W8 Calibration

Completed a fresh AutoRound 0.16.1 calibration from the original BF16 model:
W8A16, symmetric group128, 200 iterations per block, batch size 8, seed 42,
128 random 2048-token windows from WikiText-2 raw TRAIN only. Text blocks use
SignRound optimization; linear-attention in_proj_a/b remain floating point.
Vision and lm_head remain unquantized. Export additionally applies RTN to seven
MTP linear weights not exercised by text calibration; these are not used by
the ordinary text forward scored here. This is not an exact reproduction of
Intel's unpublished calibration data or its older AutoRound version.

- Script: `scripts/local/autoround_teacher.py`
- Model directory: `/data/models/Qwen3.5-0.8B-AutoRound-W8-BF16-cal128-seq2048/Qwen3.5-0.8B-w8g128`
- Sampling manifest: `/data/models/Qwen3.5-0.8B-AutoRound-W8-BF16-cal128-seq2048/teacher_experiment.json`
- Calibration log: `/data/autoround_w8_full.log`
- Tuning time: 662 seconds; reported peak VRAM 4.47 GB.
- Evaluation: `/data/anyprecision_mopd/new_teachers/autoround_w8_full/`
- Full test PPL: 19.401943473234446, 296902 scored tokens.

The local model is closer to the BF16 baseline than the downloaded axteam W8,
but does not improve on the previous GPTQ W8 result. Dtype/scope differences
remain a confounder. No MOPD teacher configuration was changed.

```bash
CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 /usr/bin/python \
  scripts/local/autoround_teacher.py \
  --output /data/models/Qwen3.5-0.8B-AutoRound-W8-repeat
```

### PPL Evaluation

Run from the repository root with a new output directory:

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 /usr/bin/python \
  scripts/local/eval_anyprecision_wikitext.py \
  --standalone-model /data/models/Qwen3.5-0.8B-int4-AutoRound \
  --quant-backend auto_round:torch_zp \
  --output /data/anyprecision_mopd/new_teachers/w4_repeat
```

For W8 use `/data/models/Qwen3.5-0.8B-AWQ-8bit` and omit `--quant-backend`.
For the baseline use `/workspace/my_project/models/Qwen3.5-0.8B` and omit
`--quant-backend`. Existing results are never overwritten.