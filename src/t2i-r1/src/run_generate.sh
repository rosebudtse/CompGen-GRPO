#!/bin/bash
# T2I-CompBench 生图脚本（多卡数据并行）
#
# Usage:
#   bash run_generate.sh --model_path <ckpt> --save_root <out> [--num_generation 10] [--nproc 2]
#
# Examples:
#   # 默认：用 2 张卡跑 checkpoint-800
#   bash run_generate.sh \
#       --model_path outputs/train_g8_1k6_full/checkpoint-800 \
#       --save_root /mlx_devbox/users/xiezifan/playground/CompGen-GRPO/eval_results/finetuned_step800
#
#   # 跑 baseline（原始 Janus-Pro-1B）
#   bash run_generate.sh \
#       --model_path /mlx_devbox/users/xiezifan/playground/CompGen-GRPO/src/t2i-r1/reward_weight/Janus-Pro-1B \
#       --save_root /mlx_devbox/users/xiezifan/playground/CompGen-GRPO/eval_results/baseline

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

# ── 屏蔽刷屏（与 run_train.sh 一致）────────────────────────────────
# FutureWarning / DeprecationWarning / UserWarning（image_processor_class、torch_dtype 等）
export PYTHONWARNINGS="ignore::FutureWarning,ignore::DeprecationWarning,ignore::UserWarning"
# transformers logger 的 INFO/WARNING（"Using a slow image processor"、"torch_dtype is deprecated"）
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
# 单节点 NCCL 优化 & 减少刷屏
export NCCL_DEBUG=WARN
export NCCL_NET_PLUGIN=none
export NCCL_IB_DISABLE=1

# 默认参数
NPROC=2
MODEL_PATH="outputs/train_g8_1k6_full/checkpoint-800"
SAVE_ROOT="/mlx_devbox/users/xiezifan/playground/CompGen-GRPO/eval_results/finetuned"
DATASET_DIR="/mlx_devbox/users/xiezifan/playground/T2I-CompBench/examples/dataset"
REASONING_PROMPT="/mlx_devbox/users/xiezifan/playground/CompGen-GRPO/data/prompt/reasoning_prompt.txt"
NUM_GENERATION=10
CFG_WEIGHT=5.0
SEED=42
SKIP_EXISTING="--skip_existing"
CATEGORIES=""  # 留空表示跑全部 6 类

while [[ $# -gt 0 ]]; do
    case $1 in
        --nproc)           NPROC="$2";          shift 2 ;;
        --model_path)      MODEL_PATH="$2";     shift 2 ;;
        --save_root)       SAVE_ROOT="$2";      shift 2 ;;
        --dataset_dir)     DATASET_DIR="$2";    shift 2 ;;
        --reasoning_prompt) REASONING_PROMPT="$2"; shift 2 ;;
        --num_generation)  NUM_GENERATION="$2"; shift 2 ;;
        --cfg_weight)      CFG_WEIGHT="$2";     shift 2 ;;
        --seed)            SEED="$2";           shift 2 ;;
        --no_skip)         SKIP_EXISTING="";    shift 1 ;;
        --categories)      CATEGORIES="$2";     shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# CUDA_VISIBLE_DEVICES 由调用者设置；这里只 print
echo ""
echo "=============================================="
echo "  T2I-CompBench Generation"
echo "  GPUs (CUDA_VISIBLE_DEVICES)  : ${CUDA_VISIBLE_DEVICES:-all}"
echo "  Processes                    : $NPROC"
echo "  Model path                   : $MODEL_PATH"
echo "  Save root                    : $SAVE_ROOT"
echo "  Dataset dir                  : $DATASET_DIR"
echo "  num_generation per prompt    : $NUM_GENERATION"
echo "  cfg_weight                   : $CFG_WEIGHT"
echo "  seed                         : $SEED"
echo "  skip_existing                : ${SKIP_EXISTING:-off}"
echo "  categories                   : ${CATEGORIES:-(all 6)}"
echo "=============================================="
echo ""

mkdir -p "$SAVE_ROOT"

# ── 把脚本所有 stdout/stderr 固化到 <save_root>/logs/gen_<时间戳>.log ──
#    同时维护一个 gen_latest.log 软链方便 tail -f
LOG_DIR="$SAVE_ROOT/logs"
mkdir -p "$LOG_DIR"
TS="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$LOG_DIR/gen_${TS}.log"
ln -sf "$(basename "$LOG_FILE")" "$LOG_DIR/gen_latest.log"
exec > >(tee "$LOG_FILE") 2>&1

echo "[run_generate.sh] LOG_FILE=$LOG_FILE"
echo "[run_generate.sh] start at $(date '+%F %T')"

CAT_ARGS=""
if [ -n "$CATEGORIES" ]; then
    CAT_ARGS="--categories $CATEGORIES"
fi

cd "$SCRIPT_DIR"

# --standalone 让 torchrun 自动选空闲端口，避免 EADDRINUSE
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NPROC \
    generate_all_eval.py \
    --model_path "$MODEL_PATH" \
    --dataset_dir "$DATASET_DIR" \
    --save_root "$SAVE_ROOT" \
    --reasoning_prompt_path "$REASONING_PROMPT" \
    --num_generation $NUM_GENERATION \
    --cfg_weight $CFG_WEIGHT \
    --seed $SEED \
    $SKIP_EXISTING \
    $CAT_ARGS

echo ""
echo "✅ Generation done. Now run:"
echo "    bash run_eval.sh --model finetuned --task all"

echo ""
echo "[run_generate.sh] finished at $(date '+%F %T') (exit=$?)"
