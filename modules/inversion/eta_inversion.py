import time
from tqdm import tqdm

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


class EtaInversion(DiffusionInversion):
    """Main class for eta inversion.
    """

    def __init__(self, model: StableDiffusionPipeline, scheduler: Optional[str]=None, num_inference_steps: Optional[int]=None, 
                 guidance_scale_bwd: Optional[float]=None, guidance_scale_fwd: Optional[float]=None,
                 verbose: bool=False, eta=(0.0, 0.4), noise_sample_count: int=10, seed: int=0, 
                 eta_start: Optional[float]=None, eta_end: Optional[float]=None, eta_zero_at: Optional[float]=None) -> None:
        """Creates a new eta inversion instance.

        Args:
            model (StableDiffusionPipeline): The diffusion model to invert. Must be Stable Diffusion for now.
            scheduler (Optional[str], optional): Name of the scheduler to invert. 
            Possbile choices are "ddim", "dpm" and "ddpm". Defaults to "ddim".
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

        num_train_steps = 1000  # train steps for diffusion model

        if isinstance(guidance_scale_fwd, (tuple, list)):
            assert len(guidance_scale_fwd) == 2
            guidance_scale_fwd = np.linspace(guidance_scale_fwd[0], guidance_scale_fwd[1], num_train_steps)

        super().__init__(model, scheduler, num_inference_steps, guidance_scale_bwd, guidance_scale_fwd, verbose)

        if eta_start is not None:
            # for gradio
            # override eta
            assert eta_end is not None
            eta = (eta_start, eta_end)
            print(eta, noise_sample_count, seed)
            
        if not isinstance(eta, (tuple, list)):
            eta = eta, eta

        # create all etas for all training steps
        etas = np.linspace(eta[0], eta[1], num_train_steps)

        # zero out etas
        if eta_zero_at is not None:
            etas[:int(eta_zero_at * num_train_steps)] = 0

        self.etas = etas
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

        return torch.randn((n, 1, 4, 64, 64), generator=generator, device=self.model.device)

    def predict_step_backward(self, latent: torch.Tensor, t: torch.Tensor, context: torch.Tensor, guidance_scale_bwd: Optional[float]=None, 
                              source_latent_prev: Optional[torch.Tensor]=None, generator: Optional[torch.Generator]=None) -> Tuple[torch.Tensor, torch.Tensor]:
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

        # update the latent based on the predicted noise with the noise schedulers
        new_latent = self.step_backward(noise_pred, t, latent, eta=eta_res["eta"], variance_noise=eta_res["variance_noise"]).prev_sample

        # direct inversion
        new_latent[:1] += eta_res["delta"]
        new_latent = new_latent.clone()

        # call controller callback to modify latent (e.g. ptp)
        new_latent = self.controller.end_step(latent=new_latent, noise_pred=noise_pred, t=t)

        return new_latent, noise_pred

    def diffusion_backward(self, latent: torch.Tensor, context: torch.Tensor, inv_result: Dict[str, Any]) -> torch.Tensor:
        generator = torch.Generator(device=self.model.device).manual_seed(self.seed)

        for i, t in enumerate(self.pbar(self.scheduler_bwd.timesteps, desc="backward")):
            # pass previous latent
            latent, noise_pred = self.predict_step_backward(latent, t, context, source_latent_prev=inv_result["latents"][-(i+2)], generator=generator)
            
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
