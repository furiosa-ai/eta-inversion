


import torch
from typing import Optional


class ControllerBase:
    """Base class for all controllers. Controllers, like PTP, modify the latent after each diffusion step.
    """

    def __init__(self) -> None:
        """Initializes a new controller object
        """
        pass

    def begin(self) -> None:
        """Called at the start of the diffusion process
        """
        pass

    def end(self) -> None:
        """Called at the end of the diffusion process
        """
        pass

    def begin_step(self, latent: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Called at the start of a diffusion step

        Args:
            latent torch.Tensor: Latent before prediction and step. Defaults to None.

        Returns:
            torch.Tensor: Updated latent
        """
        return latent

    def end_step(self, latent: torch.Tensor, noise_pred: Optional[torch.Tensor]=None, t: Optional[int]=None) -> torch.Tensor:
        """Called at the end of a diffusion step. Returns an updated latent.

        Args:
            latent (torch.Tensor): latent after scheduler step
            noise_pred (Optional[torch.Tensor], optional): noise prediction from unet. Defaults to None.
            t (Optional[int], optional): current timestep. Defaults to None.

        Returns:
            torch.Tensor: Updated latent
        """
        return latent
    
    def copy(self) -> "ControllerBase":
        """Copies the controller to a new instance. Needed for e.g., Edit

        Returns:
            ControllerBase: New controller instance
        """
        raise NotADirectoryError


class ControllerEmpty(ControllerBase):
    """Place-in controller without any effect on the diffusion procress.
    """

    def __init__(self) -> None:
        super().__init__()

    def copy(self) -> "ControllerEmpty":
        return self


class EdictController(ControllerBase):
    """Wrapper for controllers to work in Edict inversion
    """

    def __init__(self, controller: ControllerBase) -> None:
        super().__init__()
        
        # one controller for each latent pair
        self.controllers = [
            controller.copy() for _ in range(2)
        ]

        # current latent index (0 or 1)
        self.cur_latent_idx = None

    def begin(self) -> None:
        for controller in self.controllers:
            controller.begin()

    def end(self) -> None:
        for controller in self.controllers:
            controller.end()

    def begin_step(self, latent_idx: int) -> None:
        """Called at the start of a diffusion step

        Args:
            latent_idx (int): Index of current latent (0 or 1)
        """

        # execute controller assigned to latent pair
        self.cur_latent_idx = latent_idx
        self.controllers[self.cur_latent_idx].begin_step(None)

    def end_step(self, latent: torch.Tensor) -> torch.Tensor:
        return self.controllers[self.cur_latent_idx].end_step(latent=latent)
