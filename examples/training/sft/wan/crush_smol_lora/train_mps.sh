#!/bin/bash
#
# Wan T2V 1.3B LoRA training on Apple Silicon (MPS).
#
# Differences from train.sh (the CUDA recipe):
#   - single process, plain `python` — no torchrun/accelerate launch, no NCCL
#   - Accelerate backend with every parallel degree = 1 (MPS is a single unified-memory device)
#   - native (SDPA) attention — flash-attn/sageattention/xformers are CUDA-only
#   - bf16 everywhere; fp8 layerwise upcasting is unsupported on MPS
#   - PYTORCH_ENABLE_MPS_FALLBACK=1 so ops missing MPS kernels fall back to CPU instead of crashing
#
# See docs/apple_silicon.md for the full supported/unsupported matrix.

set -e -x

export PYTORCH_ENABLE_MPS_FALLBACK=1
export WANDB_MODE="offline"
export FINETRAINERS_LOG_LEVEL="INFO"

# Check the JSON files for the expected JSON format.
# training_mps.json uses a 49x320x512 bucket instead of the CUDA recipe's 49x480x832:
# at 480x832 the ~20k-token attention matmul hits PyTorch's tiled bmm path on MPS,
# which segfaults (upstream bug in torch 2.12.1, MPSNDArray encode). See docs/apple_silicon.md.
TRAINING_DATASET_CONFIG="examples/training/sft/wan/crush_smol_lora/training_mps.json"
VALIDATION_DATASET_FILE="examples/training/sft/wan/crush_smol_lora/validation.json"

# Single-device lane: Accelerate backend, all parallel degrees 1
parallel_cmd=(
  --parallel_backend accelerate
  --pp_degree 1 --dp_degree 1 --dp_shards 1 --cp_degree 1 --tp_degree 1
)

# Model arguments
model_cmd=(
  --model_name "wan"
  --pretrained_model_name_or_path "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
  --text_encoder_dtype bf16
  --transformer_dtype bf16
  --vae_dtype bf16
)

# Attention: native SDPA is the only MPS-viable provider
attention_cmd=(
  --attn_provider_training "transformer:native"
  --attn_provider_inference "transformer:native"
)

# Dataset arguments
dataset_cmd=(
  --dataset_config $TRAINING_DATASET_CONFIG
  --dataset_shuffle_buffer_size 10
  --enable_precomputation
  --precomputation_items 25
  --precomputation_once
)

# Dataloader arguments
dataloader_cmd=(
  --dataloader_num_workers 0
)

# Diffusion arguments
diffusion_cmd=(
  --flow_weighting_scheme "logit_normal"
)

# Training arguments
# Small step count for a first smoke run; raise once training is confirmed working.
training_cmd=(
  --training_type "lora"
  --seed 42
  --batch_size 1
  --train_steps 100
  --rank 32
  --lora_alpha 32
  --target_modules "blocks.*(to_q|to_k|to_v|to_out.0)"
  --gradient_accumulation_steps 1
  --gradient_checkpointing
  --checkpointing_steps 50
  --checkpointing_limit 2
  --enable_slicing
  --enable_tiling
)

# Optimizer arguments (bitsandbytes optimizers are CUDA-only; stick to adamw)
optimizer_cmd=(
  --optimizer "adamw"
  --lr 5e-5
  --lr_scheduler "constant_with_warmup"
  --lr_warmup_steps 20
  --lr_num_cycles 1
  --beta1 0.9
  --beta2 0.99
  --weight_decay 1e-4
  --epsilon 1e-8
  --max_grad_norm 1.0
)

# Validation arguments
validation_cmd=(
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 100
)

# Miscellaneous arguments
miscellaneous_cmd=(
  --tracker_name "finetrainers-wan-mps"
  --output_dir "outputs/wan-mps"
  --init_timeout 600
  --report_to "wandb"
)

python train.py \
  "${parallel_cmd[@]}" \
  "${model_cmd[@]}" \
  "${attention_cmd[@]}" \
  "${dataset_cmd[@]}" \
  "${dataloader_cmd[@]}" \
  "${diffusion_cmd[@]}" \
  "${training_cmd[@]}" \
  "${optimizer_cmd[@]}" \
  "${validation_cmd[@]}" \
  "${miscellaneous_cmd[@]}"

echo -ne "-------------------- Finished executing script --------------------\n\n"
