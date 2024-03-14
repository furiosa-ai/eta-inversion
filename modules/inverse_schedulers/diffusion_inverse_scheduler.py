import torch
from typing import Any


class DiffusionInverseScheduler:
    def __init__(self) -> None:
        pass

    def set_timesteps(self, num_inference_steps: int) -> None:
        """Set the number of inference steps

        Args:
            num_inference_steps (int): Number of inverse steps
        """
        raise NotImplementedError
    
    def step(self, noise_pred: torch.Tensor, t: int, latent: torch.Tensor, *args, **kwargs) -> Any:
        """Perform a scheduler step and update the current latent

        Args:
            noise_pred (torch.Tensor): Noise prediction from the diffusion model
            t (int): Timestep
            latent (torch.Tensor): Current latent

        Returns:
            Any: Updated latent result
        """

        raise NotImplementedError
