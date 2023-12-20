
import time
from utils.debug_utils import enable_deterministic
enable_deterministic()

import torch
from pathlib import Path
import cv2
import argparse

from modules import load_diffusion_model, load_inverter, load_editor
from modules.inversion.diffusion_inversion import DiffusionInversion
from modules import StablePreprocess, StablePostProc
from diffusers import StableDiffusionPipeline
from typing import List, Tuple

from utils.utils import add_argparse_arg



def split_to_words(prompt: str) -> List[str]:
    """Split prompt to words

    Args:
        prompt (str): Prompt to split

    Returns:
        List[str]: Words
    """

    # remove trailing dot
    if prompt[-1] == ".":
        prompt = prompt[:-1]
    return prompt.split(" ")


def get_edit_word(source_prompt: str, target_prompt: str) -> Tuple[str, str]:
    """Get word which differs in source and target prompt

    Args:
        source_prompt (str): Source prompt
        target_prompt (str): Target prompt

    Returns:
        Tuple[str, str]: Different word
    """
    source_prompt = split_to_words(source_prompt)
    target_prompt = split_to_words(target_prompt)

    if len(source_prompt) != len(target_prompt):
        return None

    diffs = [(s, t) for s, t in zip(source_prompt, target_prompt) if s != t]

    if len(diffs) != 1:
        return None

    return diffs[0]


@torch.no_grad()
def main(input: str, model: str, src_prompt: str, target_prompt: str, output: str, inv_method: str, edit_method: str, 
         scheduler: str, steps: int, guidance_scale_bwd: float, guidance_scale_fwd: float, edit_cfg: str, prec: str) -> None:
    enable_deterministic()

    input = Path(input)

    if output is None:
        # default output path
        output = str(input.parent / (input.name + "_inv" + input.suffix))

    device = "cuda"

    # load models
    ldm_stable, (preproc, postproc) = load_diffusion_model(model, device, variant=prec)

    if edit_cfg is None:
        # Using a default config for prompt-to-prompt if no edit_cfg yaml is specified
        if edit_method in ("ptp", "etaedit"):
            # Get blend word
            blended_word = get_edit_word(src_prompt, target_prompt)

            if blended_word is None:
                print("Provide a edit_cfg for prompt-to-prompt if source and target prompt differ in more than one word.")
                return

            edit_cfg = dict(
                is_replace_controller=False,
                prompts = [src_prompt, target_prompt],
                cross_replace_steps={'default_': .4,},
                self_replace_steps=0.6,
                blend_words=(((blended_word[0], ),
                            (blended_word[1], ))) if len(blended_word) else None,
                equilizer_params={
                    "words": (blended_word[1], ),
                    "values": (2, )
                } if len(blended_word) else None,
            )

            print(f"Using default ptp config:\n{edit_cfg}")
        else:
            edit_cfg = None

    # load inverter and editor module
    inverter = load_inverter(model=ldm_stable, type=inv_method, scheduler=scheduler, num_inference_steps=steps, guidance_scale_bwd=guidance_scale_bwd, guidance_scale_fwd=guidance_scale_fwd)
    editor = load_editor(inverter=inverter, type=edit_method)

    image = preproc(input)  # load and preprocess image

    t1 = time.time()
    edit_res = editor.edit(image, src_prompt, target_prompt, cfg=edit_cfg)  # edit image
    t2 = time.time()

    img_edit = postproc(edit_res["image"])  # postprocess output
    # save result
    cv2.imwrite(output, cv2.cvtColor(img_edit, cv2.COLOR_RGB2BGR))

    if "image_inv" in edit_res:
        img_inv = postproc(edit_res["image_inv"])  # postprocess output
        output_inv = Path(output)
        output_inv = output_inv.parent / (output_inv.stem + "_inv" + output_inv.suffix)
        # save result
        cv2.imwrite(str(output_inv), cv2.cvtColor(img_inv, cv2.COLOR_RGB2BGR))

    print(f"Saved result to {output}")

    print(f"Took {t2 - t1}s")


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description="Edits a single image.")
    parser.add_argument("--input", required=True, help="Path to image to invert.")
    parser.add_argument("--model", default="CompVis/stable-diffusion-v1-4", help="Diffusion Model.")
    parser.add_argument("--src_prompt", required=True, help="Prompt to use for inversion.")
    parser.add_argument("--target_prompt", required=True, help="Prompt to use for inversion.")
    parser.add_argument("--output", help="Path for output image.")
    add_argparse_arg(parser, "--inv_method")
    add_argparse_arg(parser, "--edit_method")
    parser.add_argument("--edit_cfg", help="Path to yaml file for editor configuration. Often needed for prompt-to-prompt.")
    parser.add_argument("--scheduler", help="Which scheduler to use.", choices=DiffusionInversion.get_available_schedulers())
    parser.add_argument("--steps", type=int, help="How many diffusion steps to use.")
    parser.add_argument("--guidance_scale_bwd", type=int, help="Classifier free guidance scale to use for backward diffusion (denoising).")
    parser.add_argument("--guidance_scale_fwd", type=int, help="Classifier free guidance scale to use for forward diffusion (inversion).")
    parser.add_argument("--prec", choices=["fp16", "fp32"], help="Precision for diffusion.")
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    main(**parse_args())