
import torch
import torch.nn.functional as F
from modules.utils import ptp_utils, ptp
from typing import List, Optional, Dict, Any

from .controller import ControllerBase
from .editor import ControllerBasedEditor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from modules.utils.ptp import AttentionControl
from modules.inversion.diffusion_inversion import DiffusionInversion


class PromptToPromptControllerBase(ControllerBase):
    """Prompt-to-prompt base controller, wrapping ptp attention controller
    """

    def __init__(self, model: StableDiffusionPipeline, controller: AttentionControl) -> None:
        """Initiates a new prompt-to-prompt controller

        Args:
            model (StableDiffusionPipeline): diffusion model
            controller (AttentionControl): PtP attention controller created with ptp.make_controller
        """

        super().__init__()

        self.model = model
        self.controller = controller
        self.step_idx = None

    def begin(self) -> None:
        self.step_idx = 0

    def end(self) -> None:
        # for debugging
        assert self.step_idx > 0, "Controller begin_step/end_step was not called"

    def begin_step(self, latent: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # register ptp at the begging of each diffusion step
        ptp_utils.register_attention_control(self.model, self.controller)
        return latent

    def end_step(self, latent: torch.Tensor, noise_pred: Optional[torch.Tensor]=None, t: Optional[int]=None) -> torch.Tensor:
        # update latent using ptp attention controller
        latent = self.controller.step_callback(latent)
        # unregister ptp at the begging of each diffusion step
        ptp_utils.register_attention_control(self.model, None)
        self.step_idx += 1
        return latent


class PromptToPromptController(PromptToPromptControllerBase):
    """Prompt-to-prompt controller for editing like add and replace
    """

    def __init__(self, model: StableDiffusionPipeline, source_prompt: str, target_prompt: str, **kwargs) -> None:
        """Initiates a new prompt-to-prompt controller

        Args:
            model (StableDiffusionPipeline): diffusion model
            source_prompt (str): source prompt for inversion
            target_prompt (str): target prompt for editing
        """

        self.model = model
        self.source_prompt = source_prompt
        self.target_prompt = target_prompt
        self.ptp_cfg = {**kwargs}

        if "prompts" in self.ptp_cfg:
            assert self.ptp_cfg["prompts"] == [source_prompt, target_prompt]
            self.ptp_cfg.pop("prompts")

        # create controller and pass to base class
        super().__init__(model, ptp.make_controller(model, prompts=[source_prompt, target_prompt], **self.ptp_cfg))

    def copy(self) -> "PromptToPromptController":
        # for edict
        return PromptToPromptController(self.model, self.source_prompt, self.target_prompt, **self.ptp_cfg)
    

class PromptToPromptControllerAttentionStore(PromptToPromptControllerBase):
    """Prompt-to-prompt controller to retrieve attention maps during diffusion.
    Used by NS-LPIPS to compute object mask
    """

    def __init__(self, model: StableDiffusionPipeline) -> None:
        """Initiates a new prompt-to-prompt controller

        Args:
            model (StableDiffusionPipeline): diffusion model
        """

        # use ptp's attention store
        super().__init__(model, ptp.AttentionStore())

    def get_attention_map(self, prompt: str, word: str, res: int=16, from_where: List[str]=("up", "down"), resize: Optional[int]=None) -> torch.Tensor:
        """Retrieve stored attention map for a specific word

        Args:
            prompt (str): Prompt used for diffusion
            word (str): Word to obtain attention map for
            res (int, optional): Which attention map size to fetch from UNet. Defaults to 16.
            from_where (List[str], optional): Keys to fetch from. Defaults to ("up", "down").
            resize (Optional[int], optional): Resize attention map. Defaults to None.

        Returns:
            torch.Tensor: Attention map tensor
        """

        attention_maps = ptp.aggregate_attention([None], self.controller, res, from_where, True, 0)

        mask_idx = prompt.split(' ').index(word) + 1 # +1 for start token

        attention_map = attention_maps[:, :, mask_idx]
        attention_map = attention_map[None]
        attention_map = attention_map / attention_map.max()

        if resize is not None:
            attention_map = F.interpolate(attention_map[None], (resize, resize), mode="bicubic")[0].clamp(0, 1)

        return attention_map


class PromptToPromptEditor(ControllerBasedEditor):
    """Prompt-to-prompt editor using a prompt-to-prompt controller for editing
    """

    def __init__(self, inverter: DiffusionInversion, no_source_backward: bool=False, dft_cfg: Dict[Any, str]=None) -> None:
        super().__init__(inverter, no_source_backward, dft_cfg)

    def make_controller(self, image: torch.Tensor, source_prompt: str, target_prompt: str, **kwargs) -> PromptToPromptController:
        return PromptToPromptController(model=self.inverter.model, source_prompt=source_prompt, target_prompt=target_prompt, **kwargs)
        # return PtpController(model=self.inverter.model, source_prompt=source_prompt, target_prompt=target_prompt, **kwargs)


