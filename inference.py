from diffusers import StableDiffusionPipeline
import torch
from loguru import logger 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_id', type=str, default="./runs/dog", 
                    help="path to trained model")
parser.add_argument('--prompt', type=str, default="A photo of sks dog in a bucket", 
                    help="prompt for the output image")
parser.add_argument('--output_path', type=str, default="./out.jpg", 
                    help="path to where we want to save the output image")
args = parser.parse_args()

@logger.catch 
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = StableDiffusionPipeline.from_pretrained(args.model_id,
             torch_dtype=torch.float16).to(device)

    image = pipe(args.prompt, 
                num_inference_steps=50, 
                guidance_scale=7.5).images[0]

    image.save(args.output_path)
    logger.info(f"Output image is at {args.output_path}")

if __name__ == "__main__":
    main()