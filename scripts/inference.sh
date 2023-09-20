# Set data-related variables
export MODEL_NAME="./runs/dog"
PROMPT="a photo of sks dog in times square" # Prompt for the output image
OUTPUT_DIR="./results/out.jpg"             # Where to save the output image

# Set GPU device and run inference
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --model_id="$MODEL_NAME" \
    --prompt="$PROMPT" \
    --output_path="$OUTPUT_DIR"
