# Set data-related variables
export DATA="dog"
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="dataset/training/$DATA"    # Path to instance images
export CLASS_DIR="dataset/regularization/$DATA" # Path to class images
export OUTPUT_DIR="./runs/$DATA"               # Path to save the model

# Set GPU device and launch training
CUDA_VISIBLE_DEVICES=0 python train_dreambooth.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --instance_data_dir="$INSTANCE_DIR" \
  --class_data_dir="$CLASS_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks $DATA" \
  --class_prompt="a photo of $DATA" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800