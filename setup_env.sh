#!/bin/bash
# setup_env.sh — 在 worker 节点（有 GPU）终端运行：bash setup_env.sh
# 方案：venv --system-site-packages 继承 byted-torch 2.7.1，只补缺失的包；不动 torch。
#
# 完成后激活：source <REPO>/.venv/bin/activate
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
GDINO_DIR="$REPO_ROOT/src/t2i-r1/src/utils/GroundingDINO"

echo "=== [1/5] 创建 venv（继承系统包，保留 byted-torch 2.7.1）==="
create_venv() {
  rm -rf "$VENV_DIR"   # 清掉可能残缺的旧 venv
  # 方式1：标准 venv（需 ensurepip）
  if python -m venv --system-site-packages "$VENV_DIR" 2>/dev/null && [ -x "$VENV_DIR/bin/pip" ]; then
    echo "  -> 标准 venv 创建成功"; return 0
  fi
  rm -rf "$VENV_DIR"
  # 方式2：有免密 sudo，装 python3.11-venv 后重试
  echo "  -> 缺 ensurepip，尝试 sudo apt 安装 python3.11-venv ..."
  if sudo -n apt-get update -y >/dev/null 2>&1 && sudo -n apt-get install -y python3.11-venv >/dev/null 2>&1; then
    if python -m venv --system-site-packages "$VENV_DIR" 2>/dev/null && [ -x "$VENV_DIR/bin/pip" ]; then
      echo "  -> apt 安装后 venv 创建成功"; return 0
    fi
    rm -rf "$VENV_DIR"
  fi
  # 方式3：用字节自带的 virtualenv（自带 pip，不依赖 ensurepip）
  echo "  -> 回退：使用 /opt/tiger/ss_bin/virtualenv"
  if command -v virtualenv >/dev/null 2>&1; then
    virtualenv --system-site-packages -p python "$VENV_DIR"
  else
    /opt/tiger/ss_bin/virtualenv --system-site-packages -p python "$VENV_DIR"
  fi
}

# 判断条件用 activate 是否存在（残缺 venv 会有 bin/python 但无 activate）
if [ ! -f "$VENV_DIR/bin/activate" ] || [ ! -x "$VENV_DIR/bin/pip" ]; then
  create_venv
fi
source "$VENV_DIR/bin/activate"
echo "  pip 来源: $(python -m pip --version)"
python -m pip install -U pip

echo "=== [2/5] 校验 torch / GPU 未被破坏 ==="
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA 不可用，venv 可能没继承 byted-torch！"
print(f"torch={torch.__version__} cuda_avail={torch.cuda.is_available()} ndev={torch.cuda.device_count()}")
PY

echo "=== [3/5] 安装缺失依赖（不含 torch / flash_attn）==="
# 训练核心：transformers 需 >=4.57 以支持 Qwen3VLForConditionalGeneration
pip install "transformers>=4.57.0" "trl==0.16.0" "deepspeed==0.15.4" accelerate
# VLM 4-bit 量化
pip install bitsandbytes
# HPS 奖励
pip install hpsv2
# GroundingDINO 运行期依赖
pip install addict yapf timm "opencv-python" "supervision>=0.22.0" pycocotools einops

echo "=== [4/5] 编译安装 GroundingDINO ==="
if command -v nvcc >/dev/null 2>&1 || [ -n "$CUDA_HOME" ]; then
  echo "检测到 CUDA toolkit，开始编译 _C 扩展"
  pip install -e "$GDINO_DIR" --no-build-isolation
else
  echo "!!! 警告：未找到 nvcc / CUDA_HOME，GroundingDINO 的 CUDA 扩展无法编译。"
  echo "    GDino reward 会因缺少 _C 而报错。请先设置 CUDA_HOME 指向 cu126 toolkit："
  echo "    export CUDA_HOME=/usr/local/cuda && export PATH=\$CUDA_HOME/bin:\$PATH"
  echo "    然后重跑：pip install -e $GDINO_DIR --no-build-isolation"
fi

echo "=== [5/5] 验证关键依赖导入 ==="
python - <<'PY'
mods = ["transformers","trl","deepspeed","accelerate","bitsandbytes"]
import importlib
for m in mods:
    try:
        x = importlib.import_module(m)
        print(f"[ OK ] {m}={getattr(x,'__version__','?')}")
    except Exception as e:
        print(f"[FAIL] {m}: {e}")
try:
    from transformers import Qwen3VLForConditionalGeneration
    print("[ OK ] Qwen3VLForConditionalGeneration 可用")
except Exception as e:
    print(f"[FAIL] Qwen3VLForConditionalGeneration 不可用（transformers 版本过低？）: {e}")
try:
    import groundingdino
    from groundingdino import _C
    print("[ OK ] groundingdino + _C 扩展可用")
except Exception as e:
    print(f"[WARN] groundingdino/_C: {e}")
PY

echo
echo "=== 完成。后续在同一 venv 下训练： ==="
echo "  source $VENV_DIR/bin/activate"
echo "  bash $REPO_ROOT/src/t2i-r1/src/run_train.sh"
