from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel
)
import matplotlib.pyplot as plt 
import torch
from loguru import logger 
import argparse
from transformers import (
    DPTFeatureExtractor, 
    DPTForDepthEstimation,
)
from PIL import Image 
from diffusers.utils import load_image
import numpy as np 
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained_model_name_or_path', 
    type=str, 
    default="stabilityai/stable-diffusion-xl-base-1.0", 
    help="pretrained path"
)

parser.add_argument('--pretrained_controlnet', 
    type=str, 
    default="diffusers/controlnet-depth-sdxl-1.0", 
    help="pretrained controlnet path"
)

parser.add_argument('--prompt', type=str, default="A photo of sks donald trump and a dog", 
                    help="prompt for the output image")
parser.add_argument('--output_path', type=str, default="./out.jpg", 
                    help="path to where we want to save the output image")
parser.add_argument('--controlnet_conditioning_scale', type=float, default=0.5, 
                    help="controlnet conditioning scale")
args = parser.parse_args()

def get_depth_map(
    feature_extractor, 
    depth_estimator,
    image
):
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image

@logger.catch 
@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    controlnet = ControlNetModel.from_pretrained(
        args.pretrained_controlnet,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    ).to(device)
    pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    ).to(device)
    pipeline.set_progress_bar_config(disable=True)
    #pipeline.load_lora_weights("/scratch/jy3694/dreambooth_xl/runs/test")
    depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
    feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
    image = load_image("https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/images/stormtrooper.png")
    depth_image = plt.imread('./out.jpg')[:, 1024:]
    #depth_image = get_depth_map(feature_extractor, depth_estimator, image)
    depth_image = Image.fromarray(depth_image)
    images = pipeline(
        args.prompt, 
        image=depth_image, 
        num_inference_steps=50, 
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
    ).images
    images[0]

    images[0].save(f"stormtrooper_depth_no_lora.png")
    # image = pipe(args.prompt, 
    #             num_inference_steps=50, 
    #             guidance_scale=7.5).images[0]

    # image.save(args.output_path)
    # logger.info(f"Output image is at {args.output_path}")

if __name__ == "__main__":
    main()