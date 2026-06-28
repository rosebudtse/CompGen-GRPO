#!/bin/bash
# run_train.sh
# 正式训练脚本：BiCoT-GRPO + 5个compositional reward函数
#
# 用法：
#   chmod +x run_train.sh
#   nohup bash run_train.sh > train_main.log 2>&1 &
#   tail -f train_main.log

# ── 自动定位仓库根目录（脚本位于 <REPO>/src/t2i-r1/src/run_train.sh）──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../src/t2i-r1/src
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"              # 仓库根
cd "$SCRIPT_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export DEBUG_MODE="true"
export LOG_PATH="$SCRIPT_DIR/outputs/train_main/reward_log.txt"

# ── 路径配置（统一基于 REPO_ROOT，换机器无需改）──────────────────
REWARD_WEIGHT="$REPO_ROOT/src/t2i-r1/reward_weight"
# Janus 基座：默认放在 reward_weight/Janus-Pro-1B；如用 HF 缓存可改成对应 snapshot 路径
MODEL_PATH="${MODEL_PATH:-$REWARD_WEIGHT/Janus-Pro-1B}"
HF_DATASET="$REPO_ROOT/data/geneval_and_t2i_data_final.json"
OUTPUT_DIR="$SCRIPT_DIR/outputs/train_main"

HPS_CKPT="$REWARD_WEIGHT/HPSv2.1/HPS_v2.1_compressed.pt"
GDINO_CKPT="$REWARD_WEIGHT/groundingdino_swint_ogc.pth"
GDINO_CFG="utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
VLM_CKPT="$REWARD_WEIGHT/Qwen3-VL-2B-Instruct"
# ──────────────────────────────────────────────────────────────

mkdir -p $OUTPUT_DIR

PYTHONPATH="$SCRIPT_DIR":$PYTHONPATH \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
torchrun --nproc_per_node="${NPROC:-1}" \
--nnodes="1" \
--node_rank="0" \
--master_addr="127.0.0.1" \
--master_port="12345" \
open_r1/grpo.py \
--use_vllm=false \
--deepspeed "../configs/zero2.json" \
--output_dir $OUTPUT_DIR \
--model_name_or_path $MODEL_PATH \
--dataset_name $HF_DATASET \
--max_prompt_length 256 \
--max_completion_length 512 \
--temperature 1.0 \
--num_generations 4 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 4 \
--logging_steps 1 \
--bf16=true \
--dtype bfloat16 \
--report_to tensorboard \
--gradient_checkpointing=false \
--attn_implementation sdpa \
--max_steps "${MAX_STEPS:-2000}" \
--run_name train_main \
--save_steps 500 \
--save_total_limit 2 \
--new_generations_image 1 \
--image_token_num_per_image 576 \
--cfg_weight 5 \
--reasoning_prompt_path "$REPO_ROOT/data/prompt/reasoning_prompt.txt" \
--reward_funcs hps gdino vlm_attr vlm_orm \
--beta 0 \
--tf32=true \
--learning_rate 1e-6 \
--hps_ckpt_path $HPS_CKPT \
--gdino_ckpt_path $GDINO_CKPT \
--gdino_config_path $GDINO_CFG \
--vlm_ckpt_path $VLM_CKPT \
2>&1 | tee $OUTPUT_DIR/train_log.txt
