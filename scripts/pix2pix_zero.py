import torch
from transformers import BlipForConditionalGeneration, BlipProcessor
from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionPix2PixZeroPipeline
import requests
from PIL import Image


def main2():
    captioner_id = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(captioner_id)
    model = BlipForConditionalGeneration.from_pretrained(captioner_id, torch_dtype=torch.float16, low_cpu_mem_usage=True)

    sd_model_ckpt = "CompVis/stable-diffusion-v1-4"
    pipeline = StableDiffusionPix2PixZeroPipeline.from_pretrained(
        sd_model_ckpt,
        caption_generator=model,
        caption_processor=processor,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
    pipeline.enable_model_cpu_offload()

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

    source_prompts = ["a cat sitting on the street"]
    target_prompts = ["a dog sitting on the street"]

    source_embeds = pipeline.get_embeds(source_prompts, batch_size=2)
    target_embeds = pipeline.get_embeds(target_prompts, batch_size=2)


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
