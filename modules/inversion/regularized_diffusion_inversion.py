from .diffusion_inversion import DiffusionInversion

import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from typing import Dict, List, Optional, Tuple, Union, Any

import torch.nn.functional as F
import numpy as np


class RegularizedDiffusionInversion(DiffusionInversion):
    """Regularized diffusion inversion used in pix2pix-zero 
    (https://github.com/pix2pixzero/pix2pix-zero/blob/main/src/utils/ddim_inv.py)
    """

    def __init__(self, model: StableDiffusionPipeline, scheduler: Optional[str]=None, num_inference_steps: Optional[int]=None, 
                 guidance_scale_bwd: Optional[float]=None, guidance_scale_fwd: Optional[float]=None, verbose: bool=False, 
                 lambda_ac: float = 20.0, lambda_kl: float = 20.0, num_reg_steps: int = 5, num_ac_rolls: int = 5,) -> None:
        """Creates a new regularized diffusion inversion instance.

        Args:
            model (StableDiffusionPipeline): The diffusion model to invert. Must be Stable Diffusion for now.
            scheduler (Optional[str], optional): Name of the scheduler to invert. 
            Possbile choices are "ddim", "dpm" and "ddpm". Defaults to "ddim".
            num_inference_steps (Optional[int], optional): Number of denoising steps. Usually set to 50. Defaults to None.
            guidance_scale_bwd (Optional[float], optional): Classifier-free guidance scale for backward process (denoising). Defaults to None.
            guidance_scale_fwd (Optional[float], optional): Classifier-free guidance scale for forward process (inversion). Defaults to None.
            verbose (bool, optional): If True, print debug messages. Defaults to False.
            lambda_ac (float, optional): Loss weight for autocorrelation. Defaults to 20.0.
            lambda_kl (float, optional): Loss weight for KL divergence. Defaults to 20.0.
            num_reg_steps (int, optional): How many regularization steps to perform. Defaults to 5.
            num_ac_rolls (int, optional): How many autocorrelation steps to perform. Defaults to 5.
        """

        super().__init__(model, scheduler, num_inference_steps, guidance_scale_bwd, guidance_scale_fwd, verbose)

        self.lambda_ac =lambda_ac
        self.lambda_kl = lambda_kl
        self.num_reg_steps = num_reg_steps
        self.num_ac_rolls = num_ac_rolls

    def auto_corr_loss(self, x: torch.Tensor, random_shift: bool=True, generator: Optional[torch.Generator]=None) -> torch.Tensor:
        """Computes autocorrelation loss.

        Args:
            x (torch.Tensor): Model output (predicted noise).
            random_shift (bool, optional): If random noise should be shifted randomly. Defaults to True.
            generator (Optional[torch.Generator], optional): For deterministic noise sampling. Defaults to None.

        Returns:
            torch.Tensor:  autocorrelation loss.
        """

        B,C,H,W = x.shape
        assert B==1
        x = x.squeeze(0)
        # x must be shape [C,H,W] now
        reg_loss = 0.0
        for ch_idx in range(x.shape[0]):
            noise = x[ch_idx][None, None,:,:]
            while True:
                if random_shift: roll_amount = torch.randint(0, noise.shape[2]//2, (), generator=generator).item() # randrange(noise.shape[2]//2)
                else: roll_amount = 1
                reg_loss += (noise*torch.roll(noise, shifts=roll_amount, dims=2)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=roll_amount, dims=3)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        return reg_loss
    
    def kl_divergence(self, x: torch.Tensor) -> torch.Tensor:
        """Computes KL divergence loss.

        Args:
            x (torch.Tensor): Model output (predicted noise).

        Returns:
            torch.Tensor: KL divergence loss.
        """

        _mu = x.mean()
        _var = x.var()
        return _var + _mu**2 - 1 - torch.log(_var+1e-7)
    
    @torch.enable_grad()
    def regularize_noise_pred(self, noise_pred: torch.Tensor, generator: Optional[torch.Generator]=None) -> torch.Tensor:
        """Perform regularization on predicted noise.

        Args:
            noise_pred (torch.Tensor): Model output (predicted noise).
            generator (Optional[torch.Generator], optional): Generator for deteterministic regularization. Defaults to None.

        Returns:
            torch.Tensor: Regularized noise prediction.
        """

        e_t = noise_pred
        for _outer in range(self.num_reg_steps):
            if self.lambda_ac>0:
                for _inner in range(self.num_ac_rolls):
                    _var = torch.autograd.Variable(e_t.detach().clone(), requires_grad=True)
                    l_ac = self.auto_corr_loss(_var, generator=generator)
                    l_ac.backward()
                    _grad = _var.grad.detach()/self.num_ac_rolls
                    e_t = e_t - self.lambda_ac*_grad
            if self.lambda_kl>0:
                _var = torch.autograd.Variable(e_t.detach().clone(), requires_grad=True)
                l_kld = self.kl_divergence(_var)
                l_kld.backward()
                _grad = _var.grad.detach()
                e_t = e_t - self.lambda_kl*_grad
            e_t = e_t.detach()
        noise_pred = e_t
        return noise_pred

    def predict_step_forward(self, latent: torch.Tensor, t: torch.Tensor, context: torch.Tensor, guidance_scale_fwd: Optional[float]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(0)

        guidance_scale_fwd = guidance_scale_fwd or self.guidance_scale_fwd
        guidance_scale_fwd = np.linspace(2, 1, 1000)[t.item()]

        # call controller callback (e.g. ptp)
        latent = self.controller.begin_step(latent=latent)

        # make a noise prediction using UNet
        noise_pred = self.predict_noise(latent, t, context, guidance_scale_fwd, is_fwd=True)

        # regularize
        noise_pred = self.regularize_noise_pred(noise_pred, generator=generator)

        # update the latent based on the predicted noise with the noise schedulers
        new_latent = self.step_forward(noise_pred, t, latent).prev_sample

        # call controller callback to modify latent (e.g. ptp)
        new_latent = self.controller.end_step(latent=new_latent, noise_pred=noise_pred, t=t)

        return new_latent, noise_pred
