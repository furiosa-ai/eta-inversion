
import contextlib
from .editor import Editor
from .injector import Injector
from ..inversion.edict_inversion import EdictInversion
from ..inversion.eta_inversion import EtaInversion
import torch
import numpy as np
from typing import Dict, Iterator, List, Optional, Any, Tuple, Union
from modules.inversion.diffusion_inversion import DiffusionInversion

from transformers import BlipForConditionalGeneration, BlipProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_pix2pix_zero import Pix2PixZeroL2Loss
from diffusers.models.attention import Attention
from diffusers.models.unet_2d_condition import UNet2DConditionOutput, UNet2DConditionModel


class Pix2PixZeroAttnProcessor:
    """An attention processor class to store the attention weights.
    In Pix2Pix Zero, it happens during computations in the cross-attention blocks."""

    def __init__(self, is_pix2pix_zero: bool=False, is_edict: bool=False) -> None:
        """Creates a new attention processor for Pix2Pix Zero

        Args:
            is_pix2pix_zero (bool, optional): If pix2pix zero should be enabled for this layer. Defaults to False.
            is_edict (bool, optional): If inverter is Edict (needs one attetion store per latent in the pair).
        """

        self.is_pix2pix_zero = is_pix2pix_zero
        self.is_edict = is_edict
        if self.is_pix2pix_zero:
            if not is_edict:
                self.reference_cross_attn_map = {}
            else:
                self.reference_cross_attn_map = [{} for _ in range(2)]

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor]=None,
        attention_mask: Optional[torch.Tensor]=None,
        timestep: Optional[torch.Tensor]=None,
        loss: Optional[Pix2PixZeroL2Loss]=None,
        latent_idx: Optional[int]=None,
    ) -> torch.Tensor:
        """Processes attention map

        Args:
            attn (Attention): Attention layer
            hidden_states (torch.Tensor): State for query (optionally keys, values)
            encoder_hidden_states (Optional[torch.Tensor], optional): State for keys and values. Defaults to None.
            attention_mask (Optional[torch.Tensor], optional): Attention mask. Defaults to None.
            timestep (Optional[torch.Tensor], optional): Current timestep. Defaults to None.
            loss (Optional[Pix2PixZeroL2Loss], optional): Pix2pix zero loss. Defaults to None.
            latent_indx (Optional[int], optional): For Edict. Latent index (0 or 1).
            
        Returns:
            torch.Tensor: Attention map
        """

        assert timestep is not None, "timestep must be set when using Pix2PixZeroAttnProcessor."

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

        if self.is_pix2pix_zero and timestep is not None:
            # new bookkeeping to save the attention weights.
            if not self.is_edict:
                attn_map = self.reference_cross_attn_map
            else:
                # select attention store for edict
                attn_map = self.reference_cross_attn_map[latent_idx]

            t = timestep.item()
            if loss is None:
                assert t not in attn_map, "Attention map already recorded."
                attn_map[t] = attention_probs.detach().cpu()
            # compute loss
            elif loss is not None:
                prev_attn_probs = attn_map.pop(t)
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


class Pix2PixZeroSourceTargetInjector(Injector):
    """Injector for target latent diffusion backward.
    """

    def __init__(self, inverter, cross_attention_guidance_amount: float=0.1) -> None:
        """Creates a new injector instance for overwriting diffusion inversion methods.

        Args:
            inverter (DiffusionInversion): Diffusion inversion instance to inject functions to.
            cross_attention_guidance_amount (float, optional): Cross attention guidance amount for pix2pix zero. Defaults to 0.1.
        """

        super().__init__(inverter)

        self.latent = None  # cache latent from predict_noise()
        self.cross_attention_guidance_amount = cross_attention_guidance_amount

    def predict_noise(self, latent: torch.Tensor, t: torch.Tensor, context: torch.Tensor, guidance_scale: Optional[Union[float, int]], 
                    is_fwd: bool=False, **kwargs) -> torch.Tensor:
        noise_preds = [self._predict_noise(l, t, c, guidance_scale, is_source, is_fwd, **kwargs) 
                       for l, c, is_source in zip(latent.chunk(2), context.chunk(2), [True, False])]
        
        noise_pred = torch.cat(noise_preds)

        return noise_pred

    def _predict_noise(self, latent: torch.Tensor, t: torch.Tensor, context: torch.Tensor, guidance_scale: Optional[Union[float, int]], 
                       is_source, is_fwd: bool=False, **kwargs) -> torch.Tensor:
        assert latent.shape[0] == 1 and not is_fwd, "Provide only one prompt."
        latent_idx = kwargs.pop("latent_idx", None)  # for edict

        if not is_source:
            # Prepare for cfg and grad computation
            latent = latent.detach().clone()
            latent = torch.cat([latent] * 2)
            latent.requires_grad = True

            # latent optimizer for grad guidance
            opt = torch.optim.SGD([latent], lr=self.cross_attention_guidance_amount)

            with torch.enable_grad():
                # initialize loss
                loss = Pix2PixZeroL2Loss()

                # predict the noise residual
                noise_pred = self.inverter.unet(
                    latent,
                    t,
                    encoder_hidden_states=context,
                    cross_attention_kwargs={"timestep": t, "loss": loss, "latent_idx": latent_idx},
                    **kwargs
                ).sample

                # update latent using gradient
                loss.loss.backward(retain_graph=False)
                opt.step()

            # cache the updated latent for step_backward()
            self.latent = latent

        # call original method and use the updated latent to make a new noise prediction
        noise_pred = self.inverter.predict_noise(latent.detach(), t, context, guidance_scale, is_fwd, 
                                                 cross_attention_kwargs={"timestep": t, "latent_idx": latent_idx}, **kwargs)

        return noise_pred
    
    # def step_backward(self, noise_pred: torch.Tensor, t: torch.Tensor, latent: torch.Tensor, *args, **kwargs) -> Any:
    #     return 

    def step_backward(self, noise_pred: torch.Tensor, t: torch.Tensor, latent: torch.Tensor, *args, **kwargs) -> Any:
         # load cached latent (only unconditional part)
        # if not is_source:
        if latent.shape[0] == 2:
            latent[:1] = self.latent.detach().chunk(2)[0]
            self.latent = None

        # call original method 
        return self.inverter.step_backward(noise_pred, t, latent, *args, **kwargs)


@contextlib.contextmanager
def set_attn_processors(unet: UNet2DConditionModel, **kwargs) -> None:
    """Modifies the UNet (`unet`) to perform Pix2Pix Zero optimizations by applying attention processors.

    Args:
        unet (UNet2DConditionModel): UNet model
    """

    pix2pix_zero_attn_procs = {}
    for name in unet.attn_processors.keys():
        if "attn2" in name:
            # apply pix2pix zero
            attn_proc = Pix2PixZeroAttnProcessor(is_pix2pix_zero=True, **kwargs)
        else:
            attn_proc = Pix2PixZeroAttnProcessor(is_pix2pix_zero=False, **kwargs)
        pix2pix_zero_attn_procs[name] = attn_proc

    # store old processors to revert after pix2pix zero
    attn_procs_old = {**unet.attn_processors}
    unet.set_attn_processor(pix2pix_zero_attn_procs)
    yield
    unet.set_attn_processor(attn_procs_old)


class Pix2PixZeroEditor(Editor):
    """MasaControl editor
    """

    def __init__(self, inverter: DiffusionInversion, cross_attention_guidance_amount: float=0.1, gen_caption=True) -> None:
        """Initiates a new editor object.

        Args:
            inverter (DiffusionInversion): _description_
            cross_attention_guidance_amount (float, optional): Cross attention guidance amount for pix2pix zero. Defaults to 0.1.
        """

        super().__init__()

        self.inverter = inverter
        self.model = self.inverter.model
        self.cross_attention_guidance_amount = cross_attention_guidance_amount
        self.gen_caption = gen_caption

        if self.gen_caption:
            # generates caption for inversion and source backward using blip
            captioner_id = "Salesforce/blip-image-captioning-base"
            self.caption_processor = BlipProcessor.from_pretrained(captioner_id)
            self.caption_generator = BlipForConditionalGeneration.from_pretrained(
                captioner_id, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(self.model.device)

    def construct_direction(self, source_prompts: Union[str, List[str]], target_prompts: Union[str, List[str]]) -> torch.Tensor:
        """Constructs the edit direction to steer the image generation process semantically.

        Args:
            source_prompts (Union[str, List[str]]): Source prompt(s) to compute source prompt embeddings as start point
            target_prompts (Union[str, List[str]]): Target prompt(s) to compute target prompt embeddings as end point

        Returns:
            torch.Tensor: Edit direction. Difference between mean of target prompt embeddings and mean of source prompt embeddings.
        """

        # convert to lists
        if not isinstance(source_prompts, (tuple, list)):
            source_prompts = [source_prompts]

        if not isinstance(target_prompts, (tuple, list)):
            target_prompts = [target_prompts]

        # compute mean embeddings and take difference
        src_context = [self.inverter.create_context(p, None) for p in source_prompts]
        target_context = [self.inverter.create_context(p, None) for p in target_prompts]
        return (torch.cat(target_context).mean(0) - torch.cat(src_context).mean(0)).unsqueeze(0)

    def generate_caption(self, image: torch.Tensor) -> str:
        """Generates caption for a given image.

        Args:
            image (torch.Tensor): (Batched) tensor image normalized to [-1,1].

        Returns:
            str: Generated prompt.
        """

        # beginning of generated caption
        text = "a photography of"

        # convert image to numpy and then preprocess for blip
        inputs = self.caption_processor(((image[0].cpu().numpy() + 1) * 127.5).transpose(1, 2, 0).astype(np.uint8), text, return_tensors="pt").to(
            device=self.model.device, dtype=self.caption_generator.dtype
        )

        # generate tokens
        self.caption_generator.to(self.model.device)
        outputs = self.caption_generator.generate(**inputs, max_new_tokens=128)

        # decode tokens to caption
        caption = self.caption_processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return caption

    def edit(self, image: torch.Tensor, source_prompt: str, target_prompt: str, cfg: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        assert cfg is None

        if self.gen_caption:
            # generate image caption using blip
            caption = self.generate_caption(image)
        else:
            # use provided source prompt as image caption
            caption = ""
            # caption = source_prompt

        # create source context from caption
        src_context = self.inverter.create_context(caption, negative_prompt=caption)

        # compute edit direction using source and target prompt
        edit_direction = self.construct_direction(source_prompt, target_prompt)

        # init target context from source and apply edit direction (to conditional part)
        target_context = src_context.clone()
        target_context[1:2] += edit_direction

        # diffusion inversion with caption to obtain inverse latent zT
        inv_res = self.inverter.invert(image, context=src_context, guidance_scale_fwd=1)

        # use attention processors. pass if inverter is edict or not.
        with set_attn_processors(self.model.unet, is_edict=isinstance(self.inverter, EdictInversion)):
            # source backward first to store attention maps
            # with Pix2PixZeroSourceInjector(self.inverter):
            #     _ = self.inverter.sample(inv_res, context=src_context)

            # use stored attention maps to guide target backward
            with Pix2PixZeroSourceTargetInjector(self.inverter, cross_attention_guidance_amount=self.cross_attention_guidance_amount):
                edit_res = self.inverter.sample(
                    inv_res, 
                    context=[src_context, target_context])

        # if editing failed (e.g., inverter and editor not compatible)
        if edit_res is None:
            return None

        return {
            "image_inv": edit_res["image"][0:1],  # Image from inversion
            "image": edit_res["image"][1:2],   # Edited image
            "latent_inv": edit_res["latent"][0:1],  # z0 for inverted image
            "latent": edit_res["latent"][1:2],  # z0 for edited image
        }