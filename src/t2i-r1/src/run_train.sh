#!/bin/bash
# run_train.sh
# 正式训练脚本：BiCoT-GRPO + 5个compositional reward函数
#
# 用法：
#   chmod +x run_train.sh
#   nohup bash run_train.sh > train_main.log 2>&1 &
#   tail -f train_main.log

cd /root/autodl-tmp/T2I-R1/src/t2i-r1/src

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export DEBUG_MODE="true"
export LOG_PATH="./outputs/train_main/reward_log.txt"

# ── 路径配置 ──────────────────────────────────────────────────
MODEL_PATH="/root/.cache/huggingface/hub/models--deepseek-ai--Janus-Pro-1B/snapshots/960ab33191f61342a4c60ae74d8dc356a39fafcb"
HF_DATASET="/root/autodl-tmp/T2I-R1/data/geneval_and_t2i_data_final.json"
OUTPUT_DIR="./outputs/train_main"

HPS_CKPT="/root/autodl-tmp/T2I-R1/src/t2i-r1/reward_weight/HPS_v2.1_compressed.pt"
GDINO_CKPT="/root/autodl-tmp/T2I-R1/src/t2i-r1/reward_weight/groundingdino_swint_ogc.pth"
GDINO_CFG="utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
VLM_CKPT="/root/autodl-tmp/T2I-R1/src/t2i-r1/reward_weight/Qwen3-VL-2B-Instruct"
# ──────────────────────────────────────────────────────────────

mkdir -p $OUTPUT_DIR

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES="0" \
torchrun --nproc_per_node="1" \
--nnodes="1" \
--node_rank="0" \
--master_addr="127.0.0.1" \
--master_port="12345" \
open_r1/grpo.py \
--use_vllm False \
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
--bf16 \
--torch_dtype bfloat16 \
--report_to tensorboard \
--gradient_checkpointing false \
--attn_implementation eager \
--max_steps 2000 \
--run_name train_main \
--save_steps 500 \
--save_total_limit 2 \
--new_generations_image 1 \
--image_token_num_per_image 576 \
--cfg_weight 5 \
--reasoning_prompt_path /root/autodl-tmp/T2I-R1/data/prompt/reasoning_prompt.txt \
--reward_funcs hps gdino vlm_attr vlm_orm \
--beta 0 \
--tf32 true \
--learning_rate 1e-6 \
--hps_ckpt_path $HPS_CKPT \
--gdino_ckpt_path $GDINO_CKPT \
--gdino_config_path $GDINO_CFG \
--vlm_ckpt_path $VLM_CKPT \
# --resume_from_checkpoint /root/autodl-tmp/T2I-R1/src/t2i-r1/src/outputs/train_main/checkpoint-300 \
2>&1 | tee $OUTPUT_DIR/train_log.txt
