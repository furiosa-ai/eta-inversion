import time
from tqdm import tqdm

from modules.editing.ptp_editor import PromptToPromptControllerAttentionStore
from utils.utils import log_delta
from .diffusion_inversion import DiffusionInversion
import torch.nn.functional as F

import torch
from torch import Tensor
import cv2
import numpy as np
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from typing import Dict, List, Optional, Union, Any, Tuple
from itertools import product
import torchvision


import os
os.system("rm -rf result/pie_eta_new/*")


class EtaTensor(torch.Tensor):
    # Hack to avoid exception in DDIM scheduler in eta > 0 condition

    def __init__(self, eta):
        self.eta = eta

    def __mul__(self, other: Any) -> Tensor:
        return self.eta * other

    def __gt__(self, other: Any) -> Tensor:
        return True


class ControllerAttentionStorePerStep(PromptToPromptControllerAttentionStore):
    def __init__(self, model: StableDiffusionPipeline, prompt, res, from_where, callback) -> None:
        super().__init__(model, max_size=res)
        self.callback = callback
        self.prompt = prompt
        self.res = res
        self.from_where = from_where

    def end_step(self, latent: torch.Tensor, noise_pred: Optional[torch.Tensor]=None, t: Optional[int]=None) -> torch.Tensor:
        # attn_map = self.get_attention_map("a cat sitting next to a mirror", "cat", resize=64)  # 1 64 64
        attn_maps = [self.get_attention_map(self.prompt, word, from_where=self.from_where, res=self.res, resize=64) for word in self.prompt.split(" ")]
        self.callback(attn_maps, t)

        return super().end_step(latent, noise_pred, t)


def _create_eta_func_pow(p1, p2, p=1):
    (x1, y1), (x2, y2) = p1, p2
    a = ((y2 - y1) / (x2 - x1) ** p)
    f_str = f"{a} * (t - {x1}) ** {p} + {y1}"
    f = eval("lambda t: " + f_str.replace("t", f"np.clip(t, {x1}, {x2})"))

    return f, f_str


class EtaInversion(DiffusionInversion):
    noise_sampler = None

    def __init__(self, model: StableDiffusionPipeline, scheduler: Optional[str]=None, num_inference_steps: Optional[int]=None, 
                 guidance_scale_bwd: Optional[float]=None, guidance_scale_fwd: Optional[float]=None,
                 verbose: bool=False, eta=(0.0, 0.4), noise_sample_count: int=10, seed: int=0, 
                 eta_start: Optional[float]=None, eta_end: Optional[float]=None, use_mask=True, mask_mode_cfg=None) -> None:
        """Creates a new eta inversion instance.

        Args:
            model (StableDiffusionPipeline): The diffusion model to invert. Must be Stable Diffusion for now.
            scheduler (Optional[str], optional): Name of the scheduler to invert. 
            Possibe choices are "ddim", "dpm" and "ddpm". Defaults to "ddim".
            num_inference_steps (Optional[int], optional): Number of denoising steps. Usually set to 50. Defaults to None.
            guidance_scale_bwd (Optional[float], optional): Classifier-free guidance scale for backward process (denoising). Defaults to None.
            guidance_scale_fwd (Optional[float], optional): Classifier-free guidance scale for forward process (inversion). Defaults to None.
            verbose (bool, optional): If True, print debug messages. Defaults to False.
            eta (tuple, optional): Eta range to use for sampling. Eta is linearly interpolated (from 0 to T). Defaults to (0.0, 0.4).
            noise_sample_count (int, optional): How many times to sample noise. Defaults to 10.
            seed (int, optional): Seed for deterministic noise sampling. Defaults to 0.
            eta_start (Optional[float], optional): eta_start and eta_end is same as eta. Defaults to None.
            eta_end (Optional[float], optional): eta_start and eta_end is same as eta. Defaults to None.
            eta_zero_at (Optional[float], optional): Set eta to zero after a certain number of timesteps is reached. 
            Must be between 0 (Eta unchanged) and 1 (Eta always zero). Defaults to None.
        """

        if use_mask:
            mask_mode_cfg_dft = dict(
                attn_from_where=["up", "down"],
                attn_res=16,
                mask_dirinv=None,
                mask_eta="fwd_mean",
                pow=None,
                target_dirinv=None,
                thres=0.2,
            )

            if mask_mode_cfg is None:
                mask_mode_cfg = {}

            mask_mode_cfg = {**mask_mode_cfg_dft, **mask_mode_cfg}
        else:
            mask_mode_cfg = None

        self.mask_mode_cfg = mask_mode_cfg

        num_train_steps = 1000  # train steps for diffusion model

        if isinstance(guidance_scale_fwd, (tuple, list)):
            assert len(guidance_scale_fwd) == 2
            guidance_scale_fwd = np.linspace(guidance_scale_fwd[0], guidance_scale_fwd[1], num_train_steps)

        super().__init__(model, scheduler, num_inference_steps, guidance_scale_bwd, guidance_scale_fwd, verbose)

        if eta_start is not None:
            # for gradio
            assert eta_end is not None
            eta = (eta_start, eta_end)
            print(eta, noise_sample_count, seed)

        if not isinstance(eta, (tuple, list)):
            eta = eta, eta

        num_train_steps = 1000
        if len(eta) == 3:
            f, f_str = _create_eta_func_pow(*eta)
            ts = np.linspace(0, 1, num_train_steps)
            etas = f(ts)
        else:
            if isinstance(eta[0], (tuple, list)):
                f, f_str = _create_eta_func_pow(*eta)
                ts = np.linspace(0, 1, num_train_steps)
                etas = f(ts)
            else:
                etas = np.linspace(eta[0], eta[1], num_train_steps)
                
        etas = np.clip(etas, 0, None)

        self.etas = etas
        self.attn_maps_forward = {}
        self.noise_sample_count = noise_sample_count

        self.seed = seed if seed >= 0 else None

    def sample_variance_noise(self, n: int, generator: Optional[torch.Generator]=None) -> torch.Tensor:
        """_summary_

        Args:
            n (int): How many variance noise tensors to sample.
            generator (Optional[torch.Generator], optional): Generator for deterministic sampling. Defaults to None.

        Returns:
            torch.Tensor: Stacked variance noise tensor.
        """

        return torch.randn((n, 1, 4, 64, 64), generator=generator, device=self.model.device).to(self.model.unet.dtype)


    def get_mask(self, key, mask, t, edit_word_idx):
        if self.mask_mode_cfg is not None:
            res = self.mask_mode_cfg["attn_res"]
            from_where = self.mask_mode_cfg["attn_from_where"]

            if self.mask_mode_cfg[key] == "gt":
                # mask = mask
                pass
            elif self.mask_mode_cfg[key] == "fwd":
                # edit_word_idx = source idx, target_idx
                mask = self.attn_maps_forward[t.item()][edit_word_idx[0]]
            elif self.mask_mode_cfg[key] == "fwd_mean":
                mask = self.attn_maps_forward["mean"][edit_word_idx[0]]
                # if self.mask_mode_cfg["aggr"] == "mean":
                #     mask = self.attn_maps_forward["mean"][edit_word_idx[0]]
                # else:
                #     mask = self.attn_maps_forward[t.item()][edit_word_idx[0]]
            elif self.mask_mode_cfg[key] == "bwd_source":
                mask = self.controller.get_attention_map(mask_idx=edit_word_idx[0], res=res, from_where=from_where, prompt_idx=0, num_prompts=2, resize=64) 
            elif self.mask_mode_cfg[key] == "bwd_target":
                mask = self.controller.get_attention_map(mask_idx=edit_word_idx[1], res=res, from_where=from_where, prompt_idx=1, num_prompts=2, resize=64) 
            elif self.mask_mode_cfg[key] == "bwd_source_target":
                mask_source = self.controller.get_attention_map(mask_idx=edit_word_idx[0], res=res, from_where=from_where, prompt_idx=0, num_prompts=2, resize=64) 
                mask_target = self.controller.get_attention_map(mask_idx=edit_word_idx[1], res=res, from_where=from_where, prompt_idx=1, num_prompts=2, resize=64) 
                mask = torch.maximum(mask_source, mask_target)
            elif self.mask_mode_cfg[key] is None:
                return None
            else:
                assert False

            smooth = True

            # if smooth:
            #     smoothing = GaussianSmoothing(channels=1, kernel_size=3, sigma=0.5, dim=2).cuda()
            #     mask = smoothing(mask)
            # mask = torch.stack([mask] * 4, 1)

            if self.mask_mode_cfg["thres"] is not None:
                # assert not smooth
                mask = (mask > self.mask_mode_cfg["thres"]).to(mask.dtype)

            if self.mask_mode_cfg["pow"] is not None:
                mask = torch.pow(mask, self.mask_mode_cfg["pow"])
        else:
            mask = None

        return mask

    def predict_step_backward(self, latent: torch.Tensor, t: torch.Tensor, context: torch.Tensor, guidance_scale_bwd: Optional[float]=None, 
                              source_latent_prev=None, generator=None, mask=None, edit_word_idx=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one backward diffusion steps. Makes a noise prediction using SD's UNet first and then updates the latent using the noise scheduler.

        Args:
            latent (torch.Tensor): Current latent.
            t (torch.Tensor): Timestep.
            context (torch.Tensor): Prompt embeddings.
            guidance_scale_bwd (Optional[float], optional): Guidance scale for classifier-free guidance. Set to None for default default scale. Defaults to None.
            source_latent_prev (Optional[torch.Tensor], optional): Source latent from inversion. Latent will be replaces by this. Defaults to None.
            generator (Optional[torch.Generator], optional): Generator for deterministic sampling. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: updated latent and noise prediction
        """

        guidance_scale_bwd = guidance_scale_bwd or self.guidance_scale_bwd

        # call controller callback (e.g. ptp)
        latent = self.controller.begin_step(latent=latent, t=t)

        # make a noise prediction using UNet
        noise_pred = self.predict_noise(latent, t, context, guidance_scale_bwd)

        # get best eta and variance noise
        eta_res = self.get_eta_variance_noise(source_latent_prev, latent[:1], t, noise_pred[:1], generator)

        # eta_res = self.compute_best_eta(source_latent_prev, latent[:1], t, noise_pred[:1], generator, mask=None)
        variance_noise = eta_res["variance_noise"]
        eta = torch.full_like(variance_noise, eta_res["eta"])

        if self.mask_mode_cfg is not None:
            mask_eta = self.get_mask("mask_eta", mask, t, edit_word_idx)
            mask_dirinv = self.get_mask("mask_dirinv", mask, t, edit_word_idx)

            if mask_eta is not None:
                eta = mask_eta * eta

            new_latent = self.step_backward(noise_pred, t, latent, eta=EtaTensor(eta), variance_noise=variance_noise).prev_sample

            delta = eta_res["latent_prev"][:1] - new_latent[:1]

            new_latent[:1] = new_latent[:1] + delta

            if self.mask_mode_cfg["target_dirinv"] is not None:

                if mask_dirinv is not None:
                    delta = (1 - mask_dirinv) * delta

                new_latent[1:] = new_latent[1:] + self.mask_mode_cfg["target_dirinv"] * delta

            # _save_mask(t, mask)
        else:
            new_latent = self.step_backward(noise_pred, t, latent, eta=EtaTensor(eta), variance_noise=variance_noise).prev_sample
            new_latent[:1] = eta_res["latent_prev"][:1]

        # update the latent based on the predicted noise with the noise schedulers
        # new_latent = self.step_backward(noise_pred, t, latent, eta=eta_res["eta"], variance_noise=eta_res["variance_noise"]).prev_sample

        # direct inversion
        # new_latent[:1] += eta_res["delta"]
        new_latent = new_latent.clone()

        # call controller callback to modify latent (e.g. ptp)
        new_latent = self.controller.end_step(latent=new_latent, noise_pred=noise_pred, t=t)

        return new_latent, noise_pred

    def diffusion_backward(self, latent: torch.Tensor, context: torch.Tensor, inv_result: Dict[str, Any]) -> torch.Tensor:
        generator = torch.Generator(device=self.model.device).manual_seed(self.seed)

        inv_cfg = inv_result["inv_cfg"]

        if inv_cfg is None:
            inv_cfg = {}

        mask = inv_cfg.get("mask", None)
        edit_word_idx = inv_cfg.get("edit_word_idx", None)

        if mask is not None:
            mask = F.interpolate(mask[None, None], (64, 64), mode="bilinear")[0].to(latent.dtype).to(self.model.device)

        for i, t in enumerate(self.pbar(self.scheduler_bwd.timesteps, desc="backward")):
            # pass noise loss
            latent, noise_pred = self.predict_step_backward(latent, t, context, source_latent_prev=inv_result["latents"][-(i+2)], 
                                                            generator=generator, mask=mask, edit_word_idx=edit_word_idx)
            
        return latent

    def compute_optimal_variance_noise(self, latent_prev: torch.Tensor, latent: torch.Tensor, t: int, eta: float, noise_pred: torch.Tensor) -> torch.Tensor:
        """Solves DDIM sampling equation for variance noise to obtain optimal variance noise (where delta becomes 0).

        Args:
            latent_prev (torch.Tensor): Previous latent (from inversion).
            latent (torch.Tensor): Current latent.
            t (int): Current timestep.
            eta (float): DDIM eta.
            noise_pred (torch.Tensor): Current model noise prediction.

        Returns:
            torch.Tensor: Optimal variance noise.
        """

        latent_prev_rec_no_noise = self.step_backward(
            noise_pred, t, latent, eta=eta, variance_noise=torch.zeros_like(noise_pred)).prev_sample
        variance = self.scheduler_bwd._get_variance(t, t - self.scheduler_bwd.config.num_train_timesteps // self.num_inference_steps)
        std_dev_t = eta * variance ** (0.5)
        
        noise_opt = (latent_prev - latent_prev_rec_no_noise) / std_dev_t

        return noise_opt

    def predict_noise(self, latent: torch.Tensor, t: torch.Tensor, context: torch.Tensor, guidance_scale: Optional[Union[float, int]], is_fwd: bool=False, **kwargs) -> torch.Tensor:
        latent_input = torch.cat([latent] * 2) if latent.shape[0] != context.shape[0] else latent  # needed by pix2pix
        noise_pred_uncond, noise_prediction_text = self.unet(latent_input, t, encoder_hidden_states=context, **kwargs)["sample"].chunk(2)

        if is_fwd:
            guidance_scale = self.guidance_scale_fwd
        if isinstance(guidance_scale, (tuple, list, dict, np.ndarray)):
            guidance_scale = guidance_scale[t.item()]  # get per timestep scale

        return noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

    def get_eta_variance_noise(self, latent_prev: torch.Tensor, latent: torch.Tensor, t: int, noise_pred: torch.Tensor, generator: Optional[torch.Generator]=None) -> Dict[str, Any]:
        """Retrieves eta and computes best variance noise.

        Args:
            latent_prev (torch.Tensor): Previous latent (from inversion).
            latent (torch.Tensor): Current latent.
            t (int): Current timestep.
            noise_pred (torch.Tensor): Current model noise prediction.
            generator (Optional[torch.Generator], optional): Generator for deterministic sampling. Defaults to None.

        Returns:
            Dict[str, Any]: Dict containing eta and variance noise.
        """

        # get eta for current timestep
        eta_choices = [self.etas[t.item()]]

        # sample random variance noices
        variance_noise_choices = self.sample_variance_noise(self.noise_sample_count, generator)

        # all possible choices
        choices = list(product(eta_choices, variance_noise_choices))

        assert len(eta_choices) == 1
        eta = eta_choices[0]

        # compute ideal noise
        opt_variance_noise = self.compute_optimal_variance_noise(latent_prev, latent, t, eta, noise_pred)

        # compute distance of each sampled noise to the ideal noise
        losses = torch.square(variance_noise_choices - opt_variance_noise).reshape(variance_noise_choices.shape[0], -1).mean(1)

        # select closest noise
        best_idx = torch.argmin(losses).item()

        eta, variance_noise = choices[best_idx]
        loss = losses[best_idx]

        # perform a scheduler backward step with selected eta and variance noise
        latent_prev_rec = self.step_backward(
            noise_pred, t, latent, eta=eta, variance_noise=variance_noise).prev_sample

        # difference from forward to backward
        delta = latent_prev - latent_prev_rec

        return {"eta": eta, "variance_noise": variance_noise, "delta": delta, "latent_prev": latent_prev, "latent_prev_rec": latent_prev_rec, "loss": loss}


    def invert(self, image: torch.Tensor, prompt: Optional[str]=None, context: Optional[torch.Tensor]=None, 
               guidance_scale_fwd: Optional[float]=None, inv_cfg: Optional[Dict[str, Any]]=None,) -> Dict[str, Any]:
        # generator = torch.Generator(device=self.model.device).manual_seed(0)

        if self.mask_mode_cfg is None:
            fwd_result = super().invert(image, prompt, context, guidance_scale_fwd, inv_cfg=inv_cfg)
        else:
            if inv_cfg["edit_word_idx"][0] is None or inv_cfg["edit_word_idx"][1] is None:
                return None

            self.attn_maps_forward = {}  # clear old maps
            with self.use_controller(ControllerAttentionStorePerStep(self.model, prompt, res=self.mask_mode_cfg["attn_res"], from_where=self.mask_mode_cfg["attn_from_where"], callback=(lambda attn, t: self.attn_maps_forward.update({t.item(): attn})))):
                fwd_result = super().invert(image, prompt, context, guidance_scale_fwd, inv_cfg=inv_cfg)
        
        if self.mask_mode_cfg is not None:
            attn_maps_lst = list(self.attn_maps_forward.values())
            num_words = len(attn_maps_lst[0])

            self.attn_maps_forward["mean"] = [torch.mean(torch.stack([a[word_idx] for a in attn_maps_lst]), dim=0) for word_idx in range(num_words)]

        # with self.use_controller(ControllerAttentionStorePerStep(self.model, (lambda attn, t: self.attn_maps_forward.update({t.item(): attn})))):
        # fwd_result = super().invert(image, prompt, context, guidance_scale_fwd, inv_cfg=inv_cfg)

        # ddim_latents = fwd_result["latents"]
        # eta_list = self.compute_eta_variance_all(ddim_latents, context, self.guidance_scale_bwd, generator)

        return fwd_result