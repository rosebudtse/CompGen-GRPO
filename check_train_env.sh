#!/bin/bash
# 在 worker 节点（有 GPU）终端运行：bash check_train_env.sh
# 一次性自检：GPU / Python 依赖 / 模型权重 / 数据文件 是否就绪

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RW="$REPO_ROOT/src/t2i-r1/reward_weight"
PASS="[ OK ]"; FAIL="[FAIL]"; WARN="[WARN]"

echo "============================================================"
echo " CompGen-GRPO 训练环境自检"
echo " repo: $REPO_ROOT"
echo "============================================================"

echo; echo "### 1. GPU ###"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader 2>&1
else
  echo "$FAIL nvidia-smi 不存在"
fi

echo; echo "### 2. Python & 关键依赖 ###"
echo "python: $(which python) ($(python -V 2>&1))"
python - <<'PY'
import importlib
def chk(mod, attr="__version__", req=None):
    try:
        m = importlib.import_module(mod)
        v = getattr(m, attr, "?")
        tag = "[ OK ]"
        if req and str(v) != req:
            tag = "[WARN]"
            print(f"{tag} {mod}={v} (requirements 要求 {req})")
        else:
            print(f"{tag} {mod}={v}")
    except Exception as e:
        print(f"[FAIL] {mod} 导入失败: {type(e).__name__}: {e}")

chk("torch", req="2.5.1")
import torch
print(f"       torch.cuda.is_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
chk("transformers")
chk("trl", req="0.16.0")
chk("deepspeed", req="0.15.4")
chk("flash_attn")
chk("accelerate")
chk("bitsandbytes")   # VLM 4-bit 量化需要
for mod in ["groundingdino", "janus"]:
    try:
        importlib.import_module(mod); print(f"[ OK ] {mod} 可导入")
    except Exception as e:
        print(f"[FAIL] {mod} 不可导入: {type(e).__name__}")
PY

echo; echo "### 3. 模型权重 ###"
check_path () {  # $1=路径 $2=描述
  if [ -e "$1" ]; then echo "$PASS $2: $1"; else echo "$FAIL $2 缺失: $1"; fi
}
check_path "$RW/HPSv2.1/HPS_v2.1_compressed.pt" "HPS v2.1"
check_path "$RW/groundingdino_swint_ogc.pth" "GroundingDINO"
check_path "$RW/Qwen3-VL-2B-Instruct" "Qwen3-VL-2B"
# Janus 可能在 HF 缓存或本地目录，两处都看
if ls -d ~/.cache/huggingface/hub/models--deepseek-ai--Janus-Pro-1B >/dev/null 2>&1; then
  echo "$PASS Janus-Pro-1B: HF 缓存中"
elif [ -d "$RW/Janus-Pro-1B" ]; then
  echo "$PASS Janus-Pro-1B: $RW/Janus-Pro-1B"
else
  echo "$FAIL Janus-Pro-1B 缺失（HF 缓存和 $RW 中都没有）"
fi

echo; echo "### 4. 数据 / prompt / config ###"
check_path "$REPO_ROOT/data/geneval_and_t2i_data_final.json" "训练数据"
check_path "$REPO_ROOT/data/prompt/reasoning_prompt.txt" "reasoning prompt"
check_path "$REPO_ROOT/src/t2i-r1/configs/zero2.json" "DeepSpeed 配置"

echo; echo "### 5. GroundingDINO 编译产物 ###"
if ls "$REPO_ROOT"/src/t2i-r1/src/utils/GroundingDINO/groundingdino/_C*.so >/dev/null 2>&1; then
  echo "$PASS GroundingDINO C 扩展已编译"
else
  echo "$WARN 未发现 _C*.so，可能需要 cd src/t2i-r1/src/utils/GroundingDINO && pip install -e ."
fi

echo; echo "============================================================"
echo " 自检结束。出现 [FAIL] 的项必须修复后才能训练。"
echo "============================================================"
