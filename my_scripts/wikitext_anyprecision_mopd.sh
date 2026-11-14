#!/usr/bin/env bash
# ==============================================================================
# 脚本名称: wikitext_anyprecision_mopd.sh
# 功能说明: WikiText-2 纯文本续写任务的 AnyPrecision MOPD 底层参数组装脚本。
#           负责注入学生基座模型、默认教师模型、WikiText-2 Parquet 数据路径、
#           无对话包装的纯文本续写模板 ({{ messages[0]["content"] }})、关闭规则奖励
#           与 KL 正则等基础配置，并最终转发至 Open-MOPD 的 scripts/local/mt_opd.sh。
#
# 启动示例 (在 /workspace/my_project/Open-MOPD 目录下执行):
#   1) 预览底层 mt_opd.sh 最终生成的 Python 启动命令:
#      bash my_scripts/wikitext_anyprecision_mopd.sh --dry-run
#
#   2) 单卡直接启动默认 W4/W8 WikiText-2 训练:
#      HF_HOME=/cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
#      CUDA_VISIBLE_DEVICES=2 \
#      bash my_scripts/wikitext_anyprecision_mopd.sh --run
#
#   3) 传入额外的 Hydra 覆盖参数 (需放在 `--` 之后):
#      CUDA_VISIBLE_DEVICES=2 \
#      bash my_scripts/wikitext_anyprecision_mopd.sh --run -- \
#          trainer.total_training_steps=5
# ==============================================================================
set -euo pipefail

# 当前脚本目录 (/workspace/my_project/Open-MOPD/my_scripts)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Open-MOPD 仓库根目录 (/workspace/my_project/Open-MOPD)
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# 上级项目根目录 (/workspace/my_project)，存放公共 models / data / outputs
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

# Python 解释器路径，优先继承上层启动器设置
export PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python}"
# 物理 GPU 编号，单独直接调用本脚本时默认使用 GPU 2
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
# 按精度展开前的每步原始提示词批大小，直接调用时默认为 8
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
# 最大提示词 token 长度，超过该长度的样本会被自动过滤
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
# 每条轨迹最大续写生成长度，直接影响 rollout 耗时与显存开销
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-128}"
# 默认的共享码本校准缓存目录 (上层预设脚本可通过环境变量覆盖为 parent8 orig2 缓存)
export ANYPRECISION_CACHE="${ANYPRECISION_CACHE:-${PROJECT_ROOT}/outputs/anyprecision_qwen35_08b/mlp_seed4_parent8_s0/calibration_cache/quantized/(Qwen3.5-0.8B)-w8_orig4-gc1-c4_s100_blk512}"
# 实验产物保存目录
export OUTPUT_DIR="${OUTPUT_DIR:-/data/anyprecision_mopd/wikitext_$(date +%Y%m%d_%H%M%S)}"

# 正式运行模式下检查输出目录的已存在父目录是否具备写权限
if [[ " ${*} " == *" --run "* || " ${*} " == *" --exec "* || "${LOCAL_RUN:-0}" == 1 ]]; then
    output_parent="$OUTPUT_DIR"
    while [[ ! -e "$output_parent" ]]; do
        output_parent="$(dirname "$output_parent")"
    done
    if [[ ! -d "$output_parent" || ! -w "$output_parent" ]]; then
        printf 'Output parent is not writable: %s\n' "$output_parent" >&2
        exit 2
    fi
fi

# 调用 Open-MOPD 本地多教师蒸馏入口 scripts/local/mt_opd.sh：
# - 关闭训练期验证 (test_freq=-1)，留出官方 WikiText-2 test 集用于训练后独立 PPL 评测
# - 使用原样透传的 chat_template，确保语言模型执行纯文本续写而非对话问答
# - 关闭显式 KL 惩罚与规则奖励，仅保留在线教师-学生逆向 KL 蒸馏信号
exec bash "${REPO_ROOT}/scripts/local/mt_opd.sh" \
    --model "${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen3.5-0.8B}" \
    --teacher "${W4_TEACHER:-${PROJECT_ROOT}/models/Qwen3.5-0.8B-W4A16-AutoRound-AWQ}" \
    --teacher "${W8_TEACHER:-${PROJECT_ROOT}/models/Qwen3.5-0.8B-GPTQ-MLP-W8-cal128-seq2048}" \
    --train "${TRAIN_FILE:-${PROJECT_ROOT}/data/wikitext2_mopd/train.parquet}" \
    --val "${VAL_FILE:-${PROJECT_ROOT}/data/wikitext2_mopd/train.parquet}" \
    --extra trainer.val_before_train=False \
    --extra trainer.test_freq=-1 \
    --extra trainer.save_freq=100 \
    --extra trainer.max_actor_ckpt_to_keep=1 \
    --extra trainer.resume_mode=disable \
    --extra data.dataloader_num_workers=0 \
    --extra data.seed=42 \
    --extra '+data.apply_chat_template_kwargs.chat_template="{{ messages[0][\"content\"] }}"' \
    --extra ++actor_rollout_ref.rollout.log_prob_top_k=4 \
    --extra actor_rollout_ref.actor.use_kl_loss=False \
    --extra algorithm.use_kl_in_reward=False \
    --extra ++mt_reward_model_1.model.quantization_backend=torch \
    --extra mt_reward_model_1.model.dtype=fp16 \
    --extra data.return_multi_modal_inputs=False \
    "$@"
