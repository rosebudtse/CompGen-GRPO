#!/bin/bash
# =============================================================================
# 上传 gen samples 到 HuggingFace dataset repo（tar 打包 + 增量上传）
# =============================================================================
# 策略：
#   1) 每个 "$m/$t/samples/" 打包成 tar（不压缩，png 已压过；tar 预装比 zip 稳）
#   2) 传单个 tar → 远端路径 "$m/$t/samples.tar"
#   3) 传完立即删本地 tar，避免占盘
#
# 好处：3000 个 png × 6 task × 5 model = 90k 个 http 请求 → 30 个 tar 请求
#       LFS 也友好很多（大文件反而快）。
#
# 前置：同 01_upload_checkpoints.sh
# Usage:
#   HF_USER=<你的HF用户名> bash scripts/hf_upload/02_upload_samples.sh
#
# 上传后结构：
#   <HF_USER>/CompGen-GRPO-eval-samples
#   ├── baseline/color/samples.tar
#   ├── baseline/shape/samples.tar
#   ├── ...
#   └── wo_gdino_800/complex/samples.tar
#
# 下载后解压：
#   tar -xf <model>/<task>/samples.tar
# =============================================================================

set -eu

# 关掉 xet 后端，避免 "Wrong Magic Number" shard cache 崩溃
# export HF_HUB_DISABLE_XET=1

HF_USER="${HF_USER:?环境变量 HF_USER 必须传入}"
REPO_NAME="${REPO_NAME:-CompGen-GRPO-eval-samples}"
REPO_ID="$HF_USER/$REPO_NAME"

EVAL="/mlx_devbox/users/xiezifan/playground/CompGen-GRPO/eval_results"
TAR_TMP="${TAR_TMP:-/tmp/hf_samples_tar}"
mkdir -p "$TAR_TMP"

# ── 待上传 model 列表 ────────────────────────────
MODELS=(baseline full_400 full_800 wo_gdino_400 wo_gdino_800 wo_attr_400 wo_attr_600 wo_attr_800 wo_gdino_600)
TASKS=(color shape texture spatial non_spatial complex)
MIN_SAMPLES=2850   # 允许 -5% 偏差

echo "=========================================="
echo "  Upload Gen Samples (tarred) to HF"
echo "  Repo:  $REPO_ID (private, dataset)"
echo "  TAR tmp dir: $TAR_TMP"
echo "=========================================="

# ── [1] Ensure repo exists ────────────────────
echo ""
echo "[1] Ensure repo exists (private, dataset)"
hf repo create "$REPO_ID" --repo-type dataset --private --exist-ok

# ── [2] 拉远端文件清单 ──────────────────────────
echo ""
echo "[2] Fetching remote file list from HF..."
REMOTE_FILES=$(python3 - "$REPO_ID" <<'PYEOF' 2>/dev/null || true
import sys
from huggingface_hub import HfApi
try:
    files = HfApi().list_repo_files(sys.argv[1], repo_type="dataset")
    print("\n".join(files))
except Exception:
    pass
PYEOF
)
n_remote=$(printf "%s\n" "$REMOTE_FILES" | grep -cve '^$' || true)
echo "  远端已有 $n_remote 个文件"

# 判定：$m/$t/samples.tar 是否已在远端
declare -A REMOTE_HAS
while IFS= read -r line; do
    case "$line" in
        */*/samples.tar) REMOTE_HAS["$line"]=1 ;;
    esac
done <<< "$REMOTE_FILES"

# ── [3] Upload README ────────────────────────
README="/mlx_devbox/users/xiezifan/playground/CompGen-GRPO/scripts/hf_upload/README_samples.md"
if [ -f "$README" ]; then
    echo ""
    echo "[3] Uploading README..."
    hf upload "$REPO_ID" "$README" README.md --repo-type dataset --commit-message "Update README" || true
fi

# ── [4] 循环：tar → upload → rm ─────────────────
echo ""
echo "[4] Packing (tar) and uploading samples..."
uploaded=0
skipped_remote=0
skipped_local=0
failed=0
for m in "${MODELS[@]}"; do
    if [ ! -d "$EVAL/$m" ]; then
        echo "  [SKIP-LOCAL]  $m  (本地目录不存在)"
        skipped_local=$((skipped_local+1))
        continue
    fi
    for t in "${TASKS[@]}"; do
        remote_key="$m/$t/samples.tar"
        samples_dir="$EVAL/$m/$t/samples"

        # 前置 skip：远端已有 tar
        if [ -n "${REMOTE_HAS[$remote_key]:-}" ]; then
            echo "  [SKIP-REMOTE] $remote_key  (HF 端已存在)"
            skipped_remote=$((skipped_remote+1))
            continue
        fi

        # 本地检查
        if [ ! -d "$samples_dir" ]; then
            echo "  [SKIP-LOCAL]  $m/$t  (无 samples 目录)"
            skipped_local=$((skipped_local+1))
            continue
        fi
        n=$(ls "$samples_dir"/*.png 2>/dev/null | wc -l)
        if [ "$n" -lt "$MIN_SAMPLES" ]; then
            echo "  [SKIP-LOCAL]  $m/$t  (本地仅 $n 张，不足 $MIN_SAMPLES)"
            skipped_local=$((skipped_local+1))
            continue
        fi

        tar_file="$TAR_TMP/${m}__${t}__samples.tar"
        echo ""
        echo "  → $m/$t  ($n 张)  packing..."

        # tar 不压缩，png 已压过；-C 切目录避免 tar 里出现绝对路径
        # 若 tar 已存在（上次跑到一半），直接复用
        if [ ! -f "$tar_file" ]; then
            if ! tar -cf "$tar_file" -C "$(dirname "$samples_dir")" samples; then
                echo "  [FAIL] tar 失败: $m/$t"
                rm -f "$tar_file"
                failed=$((failed+1))
                continue
            fi
        else
            echo "    (复用已存在 tar: $tar_file)"
        fi

        tar_size=$(du -sh "$tar_file" | awk '{print $1}')
        echo "    tar size: $tar_size  →  uploading as $remote_key"
        if hf upload "$REPO_ID" "$tar_file" "$remote_key" \
                --repo-type dataset \
                --commit-message "Add $remote_key ($n imgs)"; then
            uploaded=$((uploaded+1))
            rm -f "$tar_file"      # 传完删本地 tar
        else
            echo "  [FAIL] $remote_key 上传失败（tar 保留在 $tar_file，可以重跑）"
            failed=$((failed+1))
        fi
    done
done

echo ""
echo "=========================================="
echo "  Done. uploaded=$uploaded  skipped_remote=$skipped_remote  skipped_local=$skipped_local  failed=$failed"
echo "  View: https://huggingface.co/datasets/$REPO_ID"
echo "=========================================="

# ── 清理 tar tmp 目录（仅在没有残留失败 tar 时） ─────
remaining=$(find "$TAR_TMP" -maxdepth 1 -name "*.tar" 2>/dev/null | wc -l)
if [ "$remaining" -eq 0 ]; then
    rmdir "$TAR_TMP" 2>/dev/null || true
else
    echo ""
    echo "  ℹ️  $TAR_TMP 下还有 $remaining 个未上传成功的 tar，可重跑本脚本自动重传。"
fi
