import contextlib

from .editor import Editor
import torch
import numpy as np
from typing import Dict, Iterator, Optional, Any
from modules.inversion.diffusion_inversion import DiffusionInversion

from transformers import BlipForConditionalGeneration, BlipProcessor
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_pix2pix_zero import Pix2PixZeroL2Loss

from .controller import ControllerBase

import inspect


class _FunctionInject:
    def __init__(self, func, before, after) -> None:
        self.func = func
        self.before = before
        self.after = after

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.before is not None:
            out_pre = self.before(*args, **kwargs)
            out_func = self.func(*out_pre)
        else:
            out_func = self.func(**args, **kwargs)

        if self.after is not None:
            out_post = self.after(out_func)
        else:
            out_post = out_func

        return out_post


@contextlib.contextmanager
def inject_function(obj, func_name, before=None, after=None):
    func = getattr(obj, func_name)
    inj = _FunctionInject(func, before, after)
    setattr(obj, func_name, inj)
    yield
    setattr(obj, func_name, inj.func)


class Pix2PixZeroAttnProcessor:
    """An attention processor class to store the attention weights.
    In Pix2Pix Zero, it happens during computations in the cross-attention blocks."""

    def __init__(self, is_pix2pix_zero=False):
        self.is_pix2pix_zero = is_pix2pix_zero
        if self.is_pix2pix_zero:
            self.reference_cross_attn_map = {}
        self.timestep = None

    def __call__(
        self,
        attn: "Attention",
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        timestep=None,
        loss=None,
    ):
        if timestep is not None:
            self.timestep = timestep

        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        if self.is_pix2pix_zero and self.timestep is not None:
            # new bookkeeping to save the attention weights.
            if loss is None:
                self.reference_cross_attn_map[self.timestep.item()] = attention_probs.detach().cpu()
            # compute loss
            elif loss is not None:
                prev_attn_probs = self.reference_cross_attn_map.pop(self.timestep.item())
                loss.compute_loss(attention_probs, prev_attn_probs.to(attention_probs.device))
        else:
            pass
            # assert False

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
    

def prepare_unet(unet):
    """Modifies the UNet (`unet`) to perform Pix2Pix Zero optimizations."""
    pix2pix_zero_attn_procs = {}
    attn_procs = []
    for name in unet.attn_processors.keys():
        module_name = name.replace(".processor", "")
        module = unet.get_submodule(module_name)
        if "attn2" in name:
            attn_proc = Pix2PixZeroAttnProcessor(is_pix2pix_zero=True)
            pix2pix_zero_attn_procs[name] = attn_proc
            module.requires_grad_(True)
        else:
            attn_proc = Pix2PixZeroAttnProcessor(is_pix2pix_zero=False)
            pix2pix_zero_attn_procs[name] = attn_proc
            module.requires_grad_(False)
        attn_procs.append(attn_proc)

    unet.set_attn_processor(pix2pix_zero_attn_procs)
    return attn_procs


class Pix2PixZeroController(ControllerBase):
    """Prompt-to-prompt base controller, wrapping ptp attention controller
    """

    def __init__(self, model: StableDiffusionPipeline, prompt_embeds_edit, attn_procs, mode) -> None:
        """Initiates a new prompt-to-prompt controller

        Args:
            model (StableDiffusionPipeline): diffusion model
        """

        super().__init__()

        self.model = model
        self.unet = model.unet
        self.cross_attention_guidance_amount = 0.15
        self.prompt_embeds_edit = prompt_embeds_edit
        self.attn_procs = attn_procs
        self.mode = mode


    def begin_step(self, latent: torch.Tensor, t: int, *args, **kwargs) -> torch.Tensor:
        for attn_proc in self.attn_procs:
            attn_proc.timestep = t

        if self.mode == "backward_target":
            assert latent.shape[0] == 1

            x_in = latent.detach().clone()
            x_in = torch.cat([latent] * 2)  # cfg
            x_in.requires_grad = True

            # optimizer
            opt = torch.optim.SGD([x_in], lr=self.cross_attention_guidance_amount)

            with torch.enable_grad():
                # initialize loss
                loss = Pix2PixZeroL2Loss()

                # predict the noise residual
                noise_pred = self.unet(
                    x_in,
                    t,
                    encoder_hidden_states=self.prompt_embeds_edit.detach(),
                    cross_attention_kwargs={"timestep": t, "loss": loss},
                ).sample

                loss.loss.backward(retain_graph=False)
                opt.step()

            return x_in
        else:
            return latent


class Pix2PixZeroInject:
    def __init__(self, inverter) -> None:
        pass

    def step_forward(self, noise_pred, t, latent, *args, **kwargs) -> Any:
        return self.scheduler_fwd.step(noise_pred, t, latent, *args, **kwargs)

    def step_backward(self, noise_pred, t, latent, *args, **kwargs) -> Any:
        return self.scheduler_bwd.step(noise_pred, t, latent, *args, **kwargs) 


class Pix2PixZeroEditor(Editor):
    """MasaControl editor
    """

    def __init__(self, inverter: DiffusionInversion) -> None:
        """Initiates a new editor object

        Args:
            inverter (DiffusionInversion): Inverter to use for editing
            no_null_source_prompt (bool, optional): MasaControl uses an empty source prompt "" 
            for inversion by default. True overrides this behavior. Defaults to False.
        """

        super().__init__()

        self.inverter = inverter
        self.model = self.inverter.model
        self.gen_caption = True

        if self.gen_caption:
            captioner_id = "Salesforce/blip-image-captioning-base"
            self.caption_processor = BlipProcessor.from_pretrained(captioner_id)
            self.caption_generator = BlipForConditionalGeneration.from_pretrained(captioner_id, torch_dtype=torch.float16, low_cpu_mem_usage=True)

    def construct_direction(self, embs_source: torch.Tensor, embs_target: torch.Tensor):
        """Constructs the edit direction to steer the image generation process semantically."""
        return (embs_target.mean(0) - embs_source.mean(0)).unsqueeze(0)

    def generate_caption(self, images):
        """Generates caption for a given image."""
        text = "a photography of"

        prev_device = self.caption_generator.device

        inputs = self.caption_processor(((images[0].cpu().numpy() + 1) * 127.5).transpose(1, 2, 0).astype(np.uint8), text, return_tensors="pt").to(
            device=self.model.device, dtype=self.caption_generator.dtype
        )
        self.caption_generator.to(self.model.device)
        outputs = self.caption_generator.generate(**inputs, max_new_tokens=128)

        # offload caption generator
        self.caption_generator.to(prev_device)

        caption = self.caption_processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return caption

    def control_source_sample(self, attn_procs):
        for attn_proc in self.attn_procs:
            pass
            # attn_proc.timestep = t

    def edit(self, image: torch.Tensor, source_prompt: str, target_prompt: str, cfg: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        assert cfg is None

        if self.gen_caption:
            caption = self.generate_caption(image)
        else:
            assert False
            # caption = source_prompt

        # create context from prompts
        context = self.inverter.create_context(caption, negative_prompt=caption)
        src_context = self.inverter.create_context(source_prompt)
        target_context = self.inverter.create_context(target_prompt)

        # diffusion inversion with the source prompt to obtain inverse latent zT

        attn_procs = prepare_unet(self.model.unet)

        # with self.inverter.use_controller(Pix2PixZeroController(self.model, context_edit, attn_procs, mode="forward")):
        inv_res = self.inverter.invert(image, context=src_context, guidance_scale_fwd=1)

        # store attention maps from source backward
        # with self.inverter.use_controller(Pix2PixZeroController(self.model, None, attn_procs, mode="backward_source")):

        # with inject_function(self.inverter, "")
        _ = self.inverter.sample(inv_res, context=[src_context])

        edit_direction = self.construct_direction(src_context, target_context).to(self.model.device)
        context_edit = context.clone()
        context_edit[1:2] += edit_direction

        with self.inverter.use_controller(Pix2PixZeroController(self.model, context_edit, attn_procs, mode="backward_target")):
            edit_res = self.inverter.sample(inv_res, context=[context_edit])

        # TODO: unprepare unet

        return {
            "image": edit_res["image"],   # Edited image
            "latent": edit_res["latent"],  # z0 for edited image
        }
