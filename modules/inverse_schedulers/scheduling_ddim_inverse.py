
from diffusers import DDIMScheduler
from collections import namedtuple
import diffusers.schedulers.scheduling_ddim
import torch

from .diffusion_inverse_scheduler import DiffusionInverseScheduler


class DDIMInverseScheduler(DiffusionInverseScheduler):
    """Inverse DDIM scheduler for ddim inversion
    """

    # Mimic diffusers output type
    Output = namedtuple("DDIMInverseSchedulerOutput", ("prev_sample", ))

    def __init__(self, scheduler: DDIMScheduler, inv_steps: str="sameshift") -> None:
        """Creates a new DDIM inverse scheduler

        Args:
            scheduler (DDIMScheduler): DDIM scheduler to get config from
            inv_steps (str, optional): Modifies at which timestep to begin for unet and scheduler.
            Valid options are "sameshift", "samesame", "shiftshift". Defaults to "sameshift".
        """
        super().__init__()

        self.scheduler = scheduler  # original ddim scheduler
        self.is_backward = False
        self.inv_steps = inv_steps

    @staticmethod
    def from_scheduler(scheduler: DDIMScheduler, inv_steps: str="sameshift", **kwargs) -> "DDIMInverseScheduler":
        """Creates a new inverse DDIM scheduler from an existing DDIM scheduler

        Args:
            scheduler (DDIMScheduler): DDIM scheduler to get config from
            inv_steps (str, optional): Modifies at which timestep to begin for unet and scheduler.
            Valid options are "sameshift", "samesame", "shiftshift". Defaults to "sameshift".

        Returns:
            DDIMInverseScheduler: Inverse DDIM scheduler instance
        """
        return DDIMInverseScheduler(
            DDIMScheduler.from_config({**scheduler.config, **kwargs}),
            inv_steps=inv_steps,
        )

    def set_timesteps(self, num_inference_steps: int) -> None:
        self.scheduler.set_timesteps(num_inference_steps)

    @property
    def timesteps(self) -> torch.Tensor:
        """Retrieves timesteps (reverse order)

        Returns:
            torch.Tensor: scheduler timesteps
        """

        # reverse ddim scheduler steps
        steps = reversed(self.scheduler.timesteps)

        if self.scheduler.config.steps_offset != 0:
            assert steps[0] == 1

        if self.inv_steps == "shiftshift":
            # shift steps backward
            steps = [self.get_timestep(s, -1) for s in steps]

        return steps

    def ddim_step(self, sample: torch.Tensor, model_output: torch.Tensor, timestep_from: torch.Tensor, timestep_to: torch.Tensor) -> torch.Tensor:
        """Perform a ddim step from timestep_from->timestep_to. Supports both directions.

        Args:
            sample (torch.Tensor): Current latent
            model_output (torch.Tensor): Noise prediction
            timestep_from (torch.Tensor): Starting point
            timestep_to (torch.Tensor): Ending point

        Returns:
            torch.Tensor: Updated latent
        """

        # clamp timesteps at 999 (training steps)
        timestep_from = min(timestep_from, 999)
        timestep_to = min(timestep_to, 999)

        # print(torch.Tensor(timestep_from).item(), "->", torch.Tensor(timestep_to).item())

        # clamp timesteps at 0
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep_from] if timestep_from >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[timestep_to] if timestep_to >= 0 else self.scheduler.final_alpha_cumprod

        # ddim step
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * model_output
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        
        return prev_sample

    def get_timestep(self, timestep: torch.Tensor, offset: int) -> torch.Tensor:
        """Offsets current timestep by offset steps

        Args:
            timestep (torch.Tensor): Timestep to offset
            offset (int): Steps to move forward or backward

        Returns:
            torch.Tensor: Offseted timestep
        """

        return timestep + offset * (self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps)

    def step(self, noise_pred: torch.Tensor, t: int, latent: torch.Tensor) -> "DDIMInverseScheduler.Output":
        """Perform a scheduler step and update the current latent

        Args:
            noise_pred (torch.Tensor): Noise prediction from the diffusion model
            t (int): Timestep
            latent (torch.Tensor): Current latent

        Returns:
            DDIMInverseScheduler.Output: Result with updated latent
        """

        if not self.is_backward:
            if self.inv_steps == "sameshift":
                # nti implementation, start from negative timestep
                timestep_from = self.get_timestep(t, -1)
                timestep_to = t
            elif self.inv_steps in ("samesame", "shiftshift"):
                # diffusers implementation
                timestep_from = t
                timestep_to = self.get_timestep(t, +1)
            else:
                raise Exception(self.inv_steps)
        else:
            timestep_from = t
            timestep_to = self.get_timestep(t, -1)

        prev = self.ddim_step(latent, noise_pred, timestep_from, timestep_to)
        return DDIMInverseScheduler.Output(prev)