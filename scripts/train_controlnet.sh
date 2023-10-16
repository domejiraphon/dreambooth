export DATA="trump1"
export CLASS="donald trump and a dog"
export CLASS_BASE="trump1"
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export INSTANCE_DIR="/scratch/jy3694/dataset/dreambooth/training/multi/$DATA"    # Path to instance images
export CLASS_DIR="/scratch/jy3694/dataset/dreambooth/regularization/multi/$CLASS_BASE"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"
export OUTPUT_DIR="/scratch/jy3694/dreambooth_xl_controlnet/runs"

accelerate launch controlnet.py \
    --pretrained_model_name_or_path="$MODEL_NAME"  \
    --instance_data_dir="$INSTANCE_DIR" \
    --class_data_dir="$CLASS_DIR" \
    --pretrained_vae_model_name_or_path="$VAE_PATH" \
    --output_dir="$OUTPUT_DIR" \
    --with_prior_preservation \
    --prior_loss_weight=1.0 \
    --mixed_precision="fp16" \
    --instance_prompt="a photo of sks $CLASS" \
    --class_prompt="a photo of $CLASS" \
    --resolution=1024 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --learning_rate=1e-5 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --num_class_images=200 \
    --max_train_steps=800 \
    --seed="0"