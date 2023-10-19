# Refer to https://github.com/MichalGeyer/pnp-diffusers/blob/main/pnp.py

import torch
from tqdm import tqdm
from .pnp_utils import register_time, register_attention_control_efficient, register_conv_control_efficient

from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from typing import List


def get_text_embeds(model: StableDiffusionPipeline, prompt: str, negative_prompt: str, batch_size: int=1) -> torch.Tensor:
    # Tokenize text and get embeddings
    text_input = model.tokenizer(prompt, padding='max_length', max_length=model.tokenizer.model_max_length,
                                truncation=True, return_tensors='pt')
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

    # Do the same for unconditional embeddings
    uncond_input = model.tokenizer(negative_prompt, padding='max_length', max_length=model.tokenizer.model_max_length,
                                    return_tensors='pt')

    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    # Cat for final embeddings
    text_embeddings = torch.cat([uncond_embeddings] * batch_size + [text_embeddings] * batch_size)
    return text_embeddings


def denoise_step(model, x, source_latents, t, text_embeds, guidance_scale, pnp_guidance_embeds):
    # register the time step and features in pnp injection modules
    # source_latents = load_source_latents_t(t, os.path.join(model.config["latents_path"], os.path.splitext(os.path.basename(model.config["image_path"]))[0]))
    latent_model_input = torch.cat([source_latents] + ([x] * 2))

    register_time(model, t.item())

    # compute text embeddings
    text_embed_input = torch.cat([pnp_guidance_embeds, text_embeds], dim=0)

    # apply the denoising network
    noise_pred = model.unet(latent_model_input, t, encoder_hidden_states=text_embed_input)['sample']

    # perform guidance
    _, noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

    # compute the denoising step with the reference model
    denoised_latent = model.scheduler.step(noise_pred, t, x)['prev_sample']
    return denoised_latent

def init_pnp(model: StableDiffusionPipeline, conv_injection_t: int, qk_injection_t: int) -> None:
    qk_injection_timesteps = model.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
    conv_injection_timesteps = model.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
    register_attention_control_efficient(model, qk_injection_timesteps)
    register_conv_control_efficient(model, conv_injection_timesteps)

def pnp_sample(model, latents, prompt, negative_prompt, guidance_scale, pnp_f_t=0.8, pnp_attn_t=0.5):
    pnp_guidance_embeds = get_text_embeds(model, "", "").chunk(2)[0]

    n_timesteps = model.scheduler.num_inference_steps
    init_pnp(model, conv_injection_t=int(n_timesteps * pnp_f_t), qk_injection_t=int(n_timesteps * pnp_attn_t))

    text_embeds = get_text_embeds(model, prompt, negative_prompt)
    edited_img, x = sample_loop(model, latents, text_embeds=text_embeds, guidance_scale=guidance_scale, pnp_guidance_embeds=pnp_guidance_embeds)
    return edited_img, x

def decode_latent(model, latent):
    with torch.autocast(device_type='cuda', dtype=torch.float32):
        latent = 1 / 0.18215 * latent
        image = model.vae.decode(latent).sample
        # image = (image / 2 + 0.5).clamp(0, 1)
    return image

def sample_loop(model, latents, text_embeds, guidance_scale, pnp_guidance_embeds):
    x = latents[-1]

    with torch.autocast(device_type='cuda', dtype=torch.float32):
        for i, t in enumerate(tqdm(model.scheduler.timesteps, desc="Sampling")):
            x = denoise_step(model, x, latents[-(i+1)], t, text_embeds=text_embeds, guidance_scale=guidance_scale, pnp_guidance_embeds=pnp_guidance_embeds)

        decoded_latent = decode_latent(model, x)
        # T.ToPILImage()(decoded_latent[0]).save(f'{model.config["output_path"]}/output-{model.config["prompt"]}.png')
            
    return decoded_latent, x


def register_pnp(model: StableDiffusionPipeline, source_latents: List[torch.Tensor], pnp_f_t: float=0.8, pnp_attn_t: float=0.5, use_time_as_idx: bool=False) -> None:
    model.unet.forward = PnPUnetForward(model, source_latents, use_time_as_idx=use_time_as_idx)

    # pnp_guidance_embeds = get_text_embeds(model, "", "").chunk(2)[0]

    n_timesteps = model.scheduler.num_inference_steps
    init_pnp(model, conv_injection_t=int(n_timesteps * pnp_f_t), qk_injection_t=int(n_timesteps * pnp_attn_t))


def unregister_pnp(model: StableDiffusionPipeline) -> None:
    model.unet.forward = model.unet.forward.unet_forward
    register_attention_control_efficient(model, None)
    register_conv_control_efficient(model, None)


class PnPUnetForward:
    """Place-in UNet forward function using Pnp
    """

    def __init__(self, model: StableDiffusionPipeline, source_latents: List[torch.Tensor], use_time_as_idx: bool=False) -> None:
        """Initializes a new Pnp UNet forward function

        Args:
            model (StableDiffusionPipeline): SD model
            source_latents (List[torch.Tensor]): use source latents from inverse process instead of the source latents from denoising
            use_time_as_idx (bool, optional): If True, source latents are indexed by timestep, otherwise by step index. Defaults to False.
        """

        self.model = model
        self.unet_forward = model.unet.forward
        self.source_latents = source_latents
        self.idx = 0
        self.use_time_as_idx = use_time_as_idx

        self.pnp_guidance_embeds = get_text_embeds(model, "", "").chunk(2)[0]

    def __call__(self, sample: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor) -> UNet2DConditionOutput:
        register_time(self.model, timestep.item())

        # add source latents and null embeddings as first input to current sample for pnp
        latent_model_input = torch.cat([self.source_latents[-(self.idx+1) if not self.use_time_as_idx else timestep.item()], sample])
        encoder_hidden_states = torch.cat([self.pnp_guidance_embeds, encoder_hidden_states], dim=0)

        # apply the denoising network
        noise_pred = self.unet_forward(latent_model_input, timestep, encoder_hidden_states=encoder_hidden_states)['sample']

        # discard unconditional output
        noise_pred_uncond_cond = noise_pred[1:]

        self.idx += 1

        return UNet2DConditionOutput(sample=noise_pred_uncond_cond)
