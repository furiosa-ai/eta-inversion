import torch
from transformers import BlipForConditionalGeneration, BlipProcessor
from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionPix2PixZeroPipeline
import requests
from PIL import Image

from modules.models import load_diffusion_model


def generate_captions(input_prompt):
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto", torch_dtype=torch.float16)

    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to("cuda")

    outputs = model.generate(
        input_ids, temperature=0.8, num_return_sequences=16, do_sample=True, max_new_tokens=128, top_k=10
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def generate_source_target_prompts():
    source_concept = "cat"
    target_concept = "dog"

    source_text = f"Provide a caption for images containing a {source_concept}. "
    "The captions should be in English and should be no longer than 150 characters."

    target_text = f"Provide a caption for images containing a {target_concept}. "
    "The captions should be in English and should be no longer than 150 characters."

    return generate_captions(source_text), generate_captions(target_text)


def main2():
    captioner_id = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(captioner_id)
    model = BlipForConditionalGeneration.from_pretrained(captioner_id, torch_dtype=torch.float16, low_cpu_mem_usage=True).cuda()

    sd_model_ckpt = "CompVis/stable-diffusion-v1-4"
    pipeline = StableDiffusionPix2PixZeroPipeline.from_pretrained(
        sd_model_ckpt,
        caption_generator=model,
        caption_processor=processor,
        # torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda")


    # pipeline_patch = load_diffusion_model()[0]

    # # pipeline.vae=pipeline_patch.vae
    # pipeline.text_encoder=pipeline_patch.text_encoder
    # pipeline.tokenizer=pipeline_patch.tokenizer
    # pipeline.unet=pipeline_patch.unet
    # pipeline.scheduler=pipeline_patch.scheduler
    # pipeline.safety_checker=pipeline_patch.safety_checker
    # pipeline.feature_extractor=pipeline_patch.feature_extractor
    # pipeline.scheduler=pipeline_patch.scheduler
    # del pipeline_patch

    # pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    scheduler_kwargs = {
        "clip_sample": False,
        "set_alpha_to_one": False,
        }
    pipeline.scheduler = DDIMScheduler.from_config({**pipeline.scheduler.config, **scheduler_kwargs})
    pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
    # pipeline.enable_model_cpu_offload()

    img_url = "https://github.com/pix2pixzero/pix2pix-zero/raw/main/assets/test_images/cats/cat_6.png"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB").resize((512, 512))

    caption = pipeline.generate_caption(raw_image)
    # caption = ""

    generator = torch.manual_seed(0)
    inv_latents = pipeline.invert(caption, image=raw_image, generator=generator).latents

    # See the "Generating source and target embeddings" section below to
    # automate the generation of these captions with a pre-trained model like Flan-T5 as explained below.
    # source_prompts = ["a cat sitting on the street", "a cat playing in the field", "a face of a cat"]
    # target_prompts = ["a dog sitting on the street", "a dog playing in the field", "a face of a dog"]
    source_prompts, target_prompts = generate_source_target_prompts()

    # source_prompts = ["a cat sitting on the street"]
    # target_prompts = ["a dog sitting on the street"]

    source_embeds = pipeline.get_embeds(source_prompts, batch_size=2)
    target_embeds = pipeline.get_embeds(target_prompts, batch_size=2)

    import pickle
    with open("dump.pkl", "rb") as f:
        dump = pickle.load(f)

    print(caption)
    print("inv_latents", torch.mean(inv_latents).item())
    print("source_embeds", torch.mean(source_embeds).item())
    print("target_embeds", torch.mean(target_embeds).item())

    inv_latents = dump["latents"][-1]

    image = pipeline(
        caption,
        source_embeds=source_embeds,
        target_embeds=target_embeds,
        num_inference_steps=50,
        cross_attention_guidance_amount=0.15,
        generator=generator,
        latents=inv_latents,
        negative_prompt=caption,
    ).images[0]
    image.save("edited_image.png")


def main():
    pass


if __name__ == "__main__":
    main2()
