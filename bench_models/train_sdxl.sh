export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="lambdalabs/naruto-blip-captions"

export DEVICE_CNT=8

export PYTHONPATH="$PYTHONPATH:$(pwd)"

python mp_launch.py --world_size 8 bench_models/models/train_sdxl.py \
  --pretrained_model_name_or_path='stabilityai/stable-diffusion-xl-base-1.0' \
  --pretrained_vae_model_name_or_path='madebyollin/sdxl-vae-fp16-fix' \
  --dataset_name='lambdalabs/naruto-blip-captions' \
  --resolution=128 --center_crop --random_flip \
  --proportion_empty_prompts=0.2 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=10000 \
  --learning_rate=1e-06 --lr_scheduler='constant' --lr_warmup_steps=0 \
  --validation_prompt='creature' --validation_epochs 5 \
  --checkpointing_steps=5000 \
  --max_train_samples=1024 \
  --mixed_precision='no' \
  --output_dir='sdxl-naruto-model' \
  --device_cnt=8 --train_batch_size=8 --batch_cnt=8 --loop_cnt=1
