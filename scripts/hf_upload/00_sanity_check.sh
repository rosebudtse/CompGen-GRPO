#!/bin/bash
# =============================================================================
# 上传 HuggingFace 前的产物 sanity check
# =============================================================================
# 检查项：
#   1) 待传 checkpoint 目录存在，且里头有 model.safetensors / config.json
#   2) 每个 model × 每个 task 的 samples/ 有约 3000 张 png（允许 ±5%）
#   3) 报告 checkpoint 总大小 & samples 总大小
#
# 依赖：awk（预装），不用 bc。
#
# Usage:
#   bash scripts/hf_upload/00_sanity_check.sh
# =============================================================================

set -u

REPO_ROOT="/mlx_devbox/users/xiezifan/playground/CompGen-GRPO"
OUTPUTS="$REPO_ROOT/src/t2i-r1/src/outputs"
EVAL="$REPO_ROOT/eval_results"

# ── 待上传 checkpoint 列表（新增 checkpoint 加到这里）──────────────
CKPTS=(
    "ablation_wo_attr/checkpoint-400"
    "ablation_wo_attr/checkpoint-600"
    "ablation_wo_attr/checkpoint-800"
    "ablation_wo_gdino/checkpoint-600"
)

# ── 待上传 samples 的 model 列表（复用同一套） ────────────────────
MODELS=(baseline full_400 full_800 wo_gdino_400 wo_gdino_800 wo_attr_400 wo_attr_600 wo_attr_800 wo_gdino_600)
TASKS=(color shape texture spatial non_spatial complex)

EXPECTED=3000
MIN_SAMPLES=2850             # 允许 -5%
MAX_SAMPLES=6100             # full_400 的 6000 张也算合格

echo ""
echo "=========================================="
echo "  Sanity Check Before HF Upload"
echo "=========================================="

# ── Checkpoints ──────────────────────────────────
echo ""
echo "[1/2] Checkpoints"
echo "-------------------------------------------"
ckpt_total_kb=0
ckpt_ok=0
ckpt_bad=0
for c in "${CKPTS[@]}"; do
    p="$OUTPUTS/$c"
    if [ ! -d "$p" ]; then
        printf "  [MISS] %s\n" "$c"
        ckpt_bad=$((ckpt_bad+1))
        continue
    fi
    has_st=0
    for f in "$p"/*.safetensors; do
        [ -f "$f" ] && has_st=$((has_st+1))
    done
    has_cfg=0
    [ -f "$p/config.json" ] && has_cfg=1

    size_kb=$(du -sk "$p" | awk '{print $1}')
    ckpt_total_kb=$((ckpt_total_kb + size_kb))
    size_gb=$(awk -v k="$size_kb" 'BEGIN{printf "%.2f", k/1048576}')

    if [ "$has_st" -gt 0 ] && [ "$has_cfg" = 1 ]; then
        printf "  [OK]   %-40s %s GB  (safetensors=%d)\n" "$c" "$size_gb" "$has_st"
        ckpt_ok=$((ckpt_ok+1))
    else
        printf "  [BAD]  %-40s safetensors=%d config.json=%d\n" "$c" "$has_st" "$has_cfg"
        ckpt_bad=$((ckpt_bad+1))
    fi
done
ckpt_total_gb=$(awk -v k="$ckpt_total_kb" 'BEGIN{printf "%.2f", k/1048576}')
printf "  ---\n  Total: %s GB across %d checkpoints  (OK=%d, BAD=%d)\n" \
    "$ckpt_total_gb" "${#CKPTS[@]}" "$ckpt_ok" "$ckpt_bad"

# ── Samples ──────────────────────────────────────
echo ""
echo "[2/2] Gen samples (期望 $EXPECTED 张 / task，允许 $MIN_SAMPLES ~ $MAX_SAMPLES)"
echo "-------------------------------------------"
sample_total_kb=0
task_ok=0
task_bad=0
for m in "${MODELS[@]}"; do
    if [ ! -d "$EVAL/$m" ]; then
        printf "  [MISS model] %s\n" "$m"
        continue
    fi
    line="  $m"
    all_good=1
    for t in "${TASKS[@]}"; do
        n=$(ls "$EVAL/$m/$t/samples/"*.png 2>/dev/null | wc -l)
        if [ "$n" -lt "$MIN_SAMPLES" ]; then
            line+="  ${t}:${n}❌"
            all_good=0
            task_bad=$((task_bad+1))
        else
            line+="  ${t}:${n}"
            task_ok=$((task_ok+1))
        fi
    done
    size_kb=$(du -sk --exclude='logs' "$EVAL/$m" 2>/dev/null | awk '{print $1}')
    sample_total_kb=$((sample_total_kb + size_kb))
    if [ $all_good -eq 1 ]; then
        echo "$line"
    else
        echo "$line   [ready-for-upload: skip 不合格的 task]"
    fi
done
sample_total_gb=$(awk -v k="$sample_total_kb" 'BEGIN{printf "%.2f", k/1048576}')
printf "  ---\n  Total samples on disk: %s GB  (task OK=%d, BAD/UNGEN=%d)\n" \
    "$sample_total_gb" "$task_ok" "$task_bad"

# ── 汇总 ──────────────────────────────────────────
echo ""
echo "=========================================="
echo "  预估上传"
echo "=========================================="
total_gb=$(awk -v a="$ckpt_total_kb" -v b="$sample_total_kb" 'BEGIN{printf "%.2f", (a+b)/1048576}')
mins=$(awk -v g="$total_gb" 'BEGIN{printf "%d", g*1024/20/60}')   # 假设 20 MB/s
echo "  合计（可上传）      : ${total_gb} GB"
echo "  按 20 MB/s 估算耗时 : ~${mins} 分钟"
echo ""
if [ $ckpt_bad -eq 0 ] && [ $task_bad -eq 0 ]; then
    echo "  ✅ Sanity check PASS — 所有目标都完整。"
else
    echo "  ℹ️  发现部分 model/task 未完成 gen。"
    echo "     上传脚本会自动跳过它们，等 gen 完再重跑上传即可（增量）。"
fi
