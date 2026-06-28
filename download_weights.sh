#!/bin/bash
# download_weights.sh — 在 worker 节点运行：bash download_weights.sh
# HF 直连下载 4 个模型权重到 reward_weight/。
# 若需走代理：export HF_ENDPOINT=https://hf-mirror.com 再运行。
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RW="$REPO_ROOT/src/t2i-r1/reward_weight"
mkdir -p "$RW"

# 优先使用 venv 里的 hf cli
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then source "$REPO_ROOT/.venv/bin/activate"; fi
pip install -q "huggingface_hub[cli]" >/dev/null 2>&1 || true

DL() {  # 用 hf download；新旧 CLI 名称兼容
  if command -v hf >/dev/null 2>&1; then hf download "$@";
  else huggingface-cli download "$@"; fi
}

echo "=== [1/4] Janus-Pro-1B（基座模型）==="
if [ ! -f "$RW/Janus-Pro-1B/config.json" ]; then
  DL deepseek-ai/Janus-Pro-1B --local-dir "$RW/Janus-Pro-1B"
else echo "已存在，跳过"; fi

echo "=== [2/4] Qwen3-VL-2B-Instruct（VLMAttr + VLMOrm 奖励）==="
if [ ! -f "$RW/Qwen3-VL-2B-Instruct/config.json" ]; then
  DL Qwen/Qwen3-VL-2B-Instruct --local-dir "$RW/Qwen3-VL-2B-Instruct"
else echo "已存在，跳过"; fi

echo "=== [3/4] HPS v2.1 checkpoint ==="
if [ ! -f "$RW/HPS_v2.1_compressed.pt" ]; then
  # HPS v2.1 权重托管在 xswu/HPSv2 仓库
  DL xswu/HPSv2 HPS_v2.1_compressed.pt --local-dir "$RW"
else echo "已存在，跳过"; fi

echo "=== [4/4] GroundingDINO SwinT-OGC 权重 ==="
if [ ! -f "$RW/groundingdino_swint_ogc.pth" ]; then
  # 官方 release 直链
  wget -c "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth" \
       -O "$RW/groundingdino_swint_ogc.pth"
else echo "已存在，跳过"; fi

echo
echo "=== 校验 ==="
ls -lh "$RW"
echo "Janus 目录:"; ls "$RW/Janus-Pro-1B" 2>/dev/null | head
echo "Qwen3-VL 目录:"; ls "$RW/Qwen3-VL-2B-Instruct" 2>/dev/null | head
echo "完成。如有文件缺失或 0 字节，请检查网络 / HF_ENDPOINT。"
