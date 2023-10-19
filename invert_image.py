
from pathlib import Path
import cv2
import torch
import argparse
from modules import get_inversion_methods, load_inverter
from modules import StablePreprocess, StablePostProc
from diffusers import StableDiffusionPipeline

from modules.inversion.diffusion_inversion import DiffusionInversion
from utils.utils import add_argparse_arg


@torch.no_grad()
def main(input: str, prompt: str, output: str, method: str, scheduler: str, steps: int, guidance_scale_bwd: float, guidance_scale_fwd: float) -> None:
    input = Path(input)

    if output is None:
        # default output path
        output = str(input.parent / (input.name + "_inv" + input.suffix))

    device = "cuda"

    # load models
    ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)
    preproc = StablePreprocess(device, size=512, center_crop=True, return_np=False, pil_resize=True)
    postproc = StablePostProc()

    # load inverter module
    inverter = load_inverter(model=ldm_stable, type=method, scheduler=scheduler, num_inference_steps=steps, guidance_scale_bwd=guidance_scale_bwd, guidance_scale_fwd=guidance_scale_fwd)

    image = preproc(input)  # load and preprocess image
    inv_res = inverter.invert_sample(image, prompt)  # invert image
    img_inv = postproc(inv_res["image"])  # postprocess output

    # save result
    cv2.imwrite(output, cv2.cvtColor(img_inv, cv2.COLOR_RGB2BGR))
    print(f"Saved result to {output}")


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description="Inverts a single image.")
    parser.add_argument("--input", required=True, help="Path to image to invert.")
    parser.add_argument("--prompt", required=True, help="Prompt to use for inversion.")
    parser.add_argument("--output", help="Path for output image.")
    add_argparse_arg(parser, "--method")
    parser.add_argument("--scheduler", help="Which scheduler to use.", choices=DiffusionInversion.get_available_schedulers())
    parser.add_argument("--steps", type=int, help="How many diffusion steps to use.")
    parser.add_argument("--guidance_scale_bwd", type=int, help="Classifier free guidance scale to use for backward diffusion (denoising).")
    parser.add_argument("--guidance_scale_fwd", type=int, help="Classifier free guidance scale to use for forward diffusion (inversion).")
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    main(**parse_args())
