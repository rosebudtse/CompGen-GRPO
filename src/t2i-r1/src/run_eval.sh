#!/bin/bash
# T2I-CompBench Evaluation Script
#
# Usage:
#   bash run_eval.sh --model [finetuned|baseline|both] --task [color|shape|texture|spatial|non_spatial|complex|all]

#   bash run_eval.sh --model both --task all          
#   bash run_eval.sh --model finetuned --task spatial 
#   bash run_eval.sh --model both --task color        
#   bash run_eval.sh --model baseline --task complex  

BENCH_DIR="/mlx_devbox/users/xiezifan/playground/T2I-CompBench"
EVAL_ROOT="/mlx_devbox/users/xiezifan/playground/CompGen-GRPO/eval_results"

MODEL="both"
TASK="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --task)  TASK="$2";  shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── 把脚本所有 stdout/stderr 固化到 logs/eval_<model>_<task>_<时间戳>.log ──
#    同时维护一个 eval_latest.log 软链方便 tail -f
LOG_DIR="$EVAL_ROOT/logs"
mkdir -p "$LOG_DIR"
TS="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$LOG_DIR/eval_${MODEL}_${TASK}_${TS}.log"
ln -sf "$(basename "$LOG_FILE")" "$LOG_DIR/eval_latest.log"
exec > >(tee "$LOG_FILE") 2>&1

echo "[run_eval.sh] LOG_FILE=$LOG_FILE"
echo "[run_eval.sh] start at $(date '+%F %T')"

echo ""
echo "=============================================="
echo "  Model : $MODEL"
echo "  Task  : $TASK"
echo "=============================================="

run_task() {
    local MODEL_NAME=$1
    local TASK_NAME=$2
    local OUT_DIR="$EVAL_ROOT/$MODEL_NAME/$TASK_NAME"

    echo ""
    echo ">>> [$MODEL_NAME] $TASK_NAME"

    case $TASK_NAME in
        color|shape|texture)
            cd "$BENCH_DIR/BLIPvqa_eval"
            python BLIP_vqa.py --out_dir "$OUT_DIR"
            ;;
        spatial)
            cd "$BENCH_DIR/UniDet_eval"
            python 2D_spatial_eval.py --outpath "$OUT_DIR"
            ;;
        non_spatial)
            cd "$BENCH_DIR"
            python CLIPScore_eval/CLIP_similarity.py --outpath "$OUT_DIR"
            ;;
        complex)
            echo "  [Step 1/4] BLIP-VQA for complex..."
            cd "$BENCH_DIR/BLIPvqa_eval"
            python BLIP_vqa.py --out_dir "$OUT_DIR"

            echo "  [Step 2/3] UniDet for complex..."
            cd "$BENCH_DIR/UniDet_eval"
            python 2D_spatial_eval.py --outpath "$OUT_DIR"

            echo "  [Step 3/3] CLIPScore for complex..."
            cd "$BENCH_DIR"
            python CLIPScore_eval/CLIP_similarity.py --outpath "$OUT_DIR"

            echo "  [Step 4/4] 3-in-1 aggregation..."
            cd "$BENCH_DIR/3_in_1_eval"
            python 3_in_1.py --outpath "$OUT_DIR"
            ;;
        *)
            echo "  [ERROR] Unknown task: $TASK_NAME"
            return 1
            ;;
    esac

    echo "  ✅ [$MODEL_NAME/$TASK_NAME] Done"
}

if [ "$MODEL" = "both" ]; then
    MODELS=("finetuned" "baseline")
else
    MODELS=("$MODEL")
fi

if [ "$TASK" = "all" ]; then
    TASKS=("color" "shape" "texture" "spatial" "non_spatial" "complex")
else
    TASKS=("$TASK")
fi

for m in "${MODELS[@]}"; do
    for t in "${TASKS[@]}"; do
        run_task "$m" "$t"
    done
done


echo ""
echo "=============================================="
echo "  CURRENT RESULTS"
echo "=============================================="

python3 - << 'PYEOF'
import json, os

EVAL_ROOT = "/mlx_devbox/users/xiezifan/playground/CompGen-GRPO/eval_results"

RESULT_PATHS = {
    "color":       "annotation_blip/vqa_result.json",
    "shape":       "annotation_blip/vqa_result.json",
    "texture":     "annotation_blip/vqa_result.json",
    "spatial":     "labels/annotation_obj_detection_2d/vqa_result.json",
    "non_spatial": "annotation_clip/vqa_result.json",
    "complex":     "annotation_3_in_1/vqa_result.json",
}

CATEGORIES = ["color", "shape", "texture", "spatial", "non_spatial", "complex"]

def load_score(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    scores = [float(d['answer']) for d in data]
    return sum(scores) / len(scores)

sep = "-" * 48
print(f"\n  {'Category':12s}  {'Baseline':>10}  {'Finetuned':>10}  {'Delta':>8}")
print(f"  {sep}")

base_vals, fine_vals = [], []
for cat in CATEGORIES:
    rel = RESULT_PATHS[cat]
    b = load_score(f"{EVAL_ROOT}/baseline/{cat}/{rel}")
    f = load_score(f"{EVAL_ROOT}/finetuned/{cat}/{rel}")
    base_vals.append(b)
    fine_vals.append(f)

    b_str = f"{b:.4f}" if b is not None else "   N/A"
    f_str = f"{f:.4f}" if f is not None else "   N/A"
    if b is not None and f is not None:
        delta_str = f"{f-b:+.4f}"
    else:
        delta_str = "   N/A"
    print(f"  {cat:12s}  {b_str:>10}  {f_str:>10}  {delta_str:>8}")

print(f"  {sep}")
valid = [(b, f) for b, f in zip(base_vals, fine_vals) if b is not None and f is not None]
if valid:
    avg_b = sum(b for b, _ in valid) / len(valid)
    avg_f = sum(f for _, f in valid) / len(valid)
    print(f"  {'Average':12s}  {avg_b:>10.4f}  {avg_f:>10.4f}  {avg_f-avg_b:>+8.4f}")
else:
    print(f"  {'Average':12s}  {'N/A':>10}  {'N/A':>10}  {'N/A':>8}")
print()
PYEOF

echo "[run_eval.sh] finished at $(date '+%F %T') (exit=$?)"