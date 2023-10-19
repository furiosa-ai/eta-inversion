import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
from diffusers import DDIMScheduler, DiffusionPipeline, StableDiffusionPipeline  # , StableDiffusionXLPipeline
from tqdm import tqdm
import pickle


def main():
    device = "cuda"
    file = "data/eval/plug_and_play/imagenetr-fake-ti2i/imnetr-fake-ti2i.yaml"
    output_dir = Path("data/eval/plug_and_play/imagenetr-fake-ti2i/")
    img_output_dir = output_dir / "imgs"

    img_output_dir.mkdir(parents=True, exist_ok=True)

    with open(file, "r") as f:
        data = yaml.safe_load(f)

    scheduler, model = None, None

    latents_all = {}

    idx = 0
    out_data = []
    for source_prompt_idx, sample in enumerate(tqdm(data)):
        guidance_scale = sample["scale"]
        seed = sample["seed"]
        ddim_steps = sample["ddim_steps"]
        source_prompt = sample["source_prompt"]

        img_file = img_output_dir / (source_prompt + ".png")

        if not img_file.is_file():
            if model is None:
                scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
                model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=scheduler).to(device)

            latents = torch.randn([1, 4, 64, 64], generator=torch.Generator(device).manual_seed(seed), device=device)
            latents_all[source_prompt] = latents.cpu().numpy()

            res = model(source_prompt, guidance_scale=guidance_scale, num_inference_steps=ddim_steps, generator=torch.Generator(device).manual_seed(seed))
            img = res.images[0]
            img.save(img_file)

        for target_prompt in sample["target_prompts"][:3]:
            source_prompt_words = source_prompt.split(" ")
            target_prompt_words = target_prompt.split(" ")

            assert len(source_prompt_words) == len(target_prompt_words)

            words_diff = [[s, t] for s, t in zip(source_prompt_words, target_prompt_words) if s != t]
            words_diff = [[s, t] for s, t in words_diff if sorted([s, t]) not in (["a", "an"], )]

            assert len(words_diff) == 1

            out_data.append({
                "idx": idx,
                "source_prompt_idx": source_prompt_idx,
                "source_prompt": source_prompt,
                "target_prompt": target_prompt,
                "edit": [
                    ["replace", words_diff[0]]
                ],
                "seed": seed,
                "guidance_scale": guidance_scale,
                "ddim_steps": ddim_steps
            })

            idx += 1

    with open(output_dir / "prompts.yaml", "w") as f:
        yaml.dump(out_data, f)

    with open(output_dir / "latents.pkl", "wb") as f:
        pickle.dump(latents_all, f)


if __name__ == "__main__":
    main()
