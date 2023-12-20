from typing import Optional
import torch
from diffusers import DDIMScheduler
from collections import namedtuple

from .diffusion_inverse_scheduler import DiffusionInverseScheduler


class DDPMInverseScheduler(DiffusionInverseScheduler):
    """Inverse DDPM scheduler for DDPM inversion. Noises latent and computes noisemaps as 
    difference between random sampled intermediate latents and UNet noise predictions
    """

    # step output
    Output = namedtuple("DDPMInverseSchedulerOutput", ("prev_sample", "variance_noise"))

    def __init__(self, scheduler: DDIMScheduler, inv_steps: str="sameshift", eta: int=1, markovian_forward=False) -> None:
        """Creates a new DDPM inverse scheduler

        Args:
            scheduler (DDIMScheduler):  DDIM scheduler to get config from
            inv_steps (str, optional): Only sameshift supported for now. Defaults to "sameshift".
            eta (int, optional): Unused for now. Defaults to 1.
            markovian_forward (bool, optional): If True, x_t is sampled from x_t-1, otherwise x_t is sampled from x_0. Defaults to False.
        """

        super().__init__()
        self.scheduler = scheduler
        self.inv_steps = inv_steps
        self.etas = None
        self.t_to_idx = None
        self.markovian_forward = markovian_forward

    @staticmethod
    def from_scheduler(scheduler, inv_steps="sameshift", markovian_forward=False, **kwargs) -> "DDPMInverseScheduler":
        """Creates a new DDPM inverse scheduler

        Args:
            scheduler (DDIMScheduler):  DDIM scheduler to get config from
            inv_steps (str, optional): Only sameshift supported for now. Defaults to "sameshift".
            markovian_forward (bool, optional): If True, x_t is sampled from x_t-1, otherwise x_t is sampled from x_0. Defaults to False.

        Returns:
            DDPMInverseScheduler: Inverse DDPM scheduler instance
        """

        return DDPMInverseScheduler(
            DDIMScheduler.from_config({**scheduler.config, **kwargs}),
            inv_steps=inv_steps,
            markovian_forward=markovian_forward,
        )

    def set_timesteps(self, num_inference_steps):
        self.scheduler.set_timesteps(num_inference_steps)
        self.t_to_idx = {int(v):k for k,v in enumerate(self.scheduler.timesteps)}
        # self.etas = [1] * num_inference_steps
        self.etas = [1.0] * num_inference_steps
        # assert len(self.timesteps) == num_inference_steps, "setting timesteps not supported right now"

    @property
    def timesteps(self) -> torch.Tensor:
        steps = list(reversed(self.scheduler.timesteps))
        return steps

    def get_variance(self, timestep: int) -> torch.Tensor:
        """Get variance for the specified timestep

        Args:
            timestep (int): timestep to obtain variance for

        Returns:
            torch.Tensor: variance
        """

        # compute variance
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = (beta_prod_t_prev /
                    beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance

    def sample_latents(self, latent: torch.Tensor, generator: Optional[torch.Generator]=None) -> torch.Tensor:
        """Gradually add random gaussian noise to latent. Returns all intermediate latents in the process (including input latent)

        Args:
            latent (torch.Tensor): Latent to add noise to
            generator (Optional[torch.Generator], optional): Provide generator for deterministic noising. Defaults to None.

        Returns:
            torch.Tensor: Tensor containing all intermediate latents (including input latent). In reverse order.
        """
        

        num_inference_steps = len(self.timesteps)

        alpha_bar = self.scheduler.alphas_cumprod
        sqrt_one_minus_alpha_bar = (1-alpha_bar) ** 0.5
        # alphas = self.scheduler.alphas
        variance_noise_shape = (num_inference_steps, *latent.shape[1:])
        
        timesteps = self.scheduler.timesteps.to(latent.device)

        # allocate tensor for all intermediate results
        xts = torch.zeros(variance_noise_shape).to(latent.device)

        cur_latent = latent
        for t in reversed(timesteps):
            # for every timestep, sample random gaussian noise and add it to the latent according to the noise schedule
            idx = self.t_to_idx[int(t)]
            r = torch.randn(latent.shape, device=latent.device, generator=generator)

            if not self.markovian_forward:
                # sample from x_0
                xts[idx] = latent * (alpha_bar[t]**0.5) + r * sqrt_one_minus_alpha_bar[t]
            else:
                # sample from x_t-1
                t_prev = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
                alpha_bar_t = alpha_bar[t]
                alpha_bar_t_prev = alpha_bar[t_prev] if t_prev >= 0 else 1
                cur_latent = cur_latent * ((alpha_bar_t / alpha_bar_t_prev)**0.5) + r * ((1 - (alpha_bar_t / alpha_bar_t_prev))**0.5)
                xts[idx] = cur_latent

        xts = torch.cat([xts, latent],dim = 0)

        return xts

    def get_sampled_latent_by_t(self, xts: torch.Tensor, t: int) -> torch.Tensor:
        """Retrieve intermediate latent by timestep

        Args:
            xts (torch.Tensor): Intermediate latents
            t (int): Timestep to use for retrieving

        Returns:
            torch.Tensor: Retrieved intermediate latent
        """

        return xts[self.t_to_idx[int(t)]][None]

    def get_eta_by_t(self, t: int) -> float:
        """Retrieve eta by timestep (unused for now)

        Args:
            t (int): Timestep to use for retrieving

        Returns:
            float: Retrieved eta
        """

        return self.etas[self.t_to_idx[int(t)]]

    def step(self, noise_pred: torch.Tensor, t: int, latent: torch.Tensor, xts: torch.Tensor) -> "DDPMInverseScheduler.Output":
        """Perform a scheduler step and update the current latent

        Args:
            noise_pred (torch.Tensor): Noise prediction from the diffusion model
            t (int): Timestep
            latent (torch.Tensor): Current latent (unused)
            xts (torch.Tensor): Sampled intermediate latents

        Returns:
            DDPMInverseScheduler.Output: New latent and noisemap
        """

        alpha_bar = self.scheduler.alphas_cumprod
        idx = self.t_to_idx[int(t)]

        # 1. predict noise residual
        eta = self.etas[idx]
        xt = xts[idx][None]

        xtm1 = xts[idx + 1][None]
        # pred of x0
        pred_original_sample = (
            xt - (1 - alpha_bar[t])**0.5 * noise_pred) / alpha_bar[t]**0.5

        # direction to xt
        prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod

        variance = self.get_variance(t)
        pred_sample_direction = (1 - alpha_prod_t_prev -
                                    eta * variance)**(0.5) * noise_pred

        mu_xt = alpha_prod_t_prev**(
            0.5) * pred_original_sample + pred_sample_direction

        z = (xtm1 - mu_xt) / (eta * variance**0.5)

        # correction to avoid error accumulation
        xtm1 = mu_xt + (eta * variance**0.5) * z
        # xts[idx + 1] = xtm1
    
        return DDPMInverseScheduler.Output(xtm1, z)
        # return z, xtm1
        # zs[idx] = z
        # xts[idx + 1] = xtm1
        # xts_by_time[t.item()] = xtm1
