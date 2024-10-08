
from pathlib import Path
import torch
import cv2
import numpy as np
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
        # assert self.step_idx > 0, "Controller begin_step/end_step was not called"
        pass

    def get_attention_map(self, prompt: str=None, word: str=None, res: int=16, from_where: List[str]=("up", "down"), resize: Optional[int]=None, prompt_idx=None, num_prompts=None, mask_idx=None) -> torch.Tensor:
        """Retrieve stored attention map for a specific word

        Args:
            prompt (str): Prompt used for diffusion
            word (str): Word to obtain attention map for
            res (int, optional): Which attention map size to fetch from UNet. Defaults to 16.
            from_where (List[str], optional): Keys to fetch from. Defaults to ("up", "down").
            resize (Optional[int], optional): Resize attention map. Defaults to None.
            prompt_idx (int, optional): For which prompt in the batch to retrieve attention map (e.g., 0 for source, 1 for target). Defaults to 0.
            
        Returns:
            torch.Tensor: Attention map tensor
        """

        if prompt_idx is None:
            assert num_prompts is None
            prompt_idx = 0
            num_prompts = 1
        else:
            assert num_prompts is not None

        attention_maps = ptp.aggregate_attention([None] * num_prompts, self.controller, res, from_where, True, select=prompt_idx)
        # h = 8  # only get source attention map
        # attention_maps = attention_maps[8:]
        # attention_maps = attention_maps[h*(prompt_idx):h*(prompt_idx+1)]

        if mask_idx is None:
            try:
                mask_idx = prompt.split(' ').index(word) + 1 # +1 for start token
            except ValueError:
                raise Exception(f"Cannot get attention map. Word {word} not in {prompt}")
        else:
            mask_idx = mask_idx + 1

        attention_map = attention_maps[:, :, mask_idx]
        attention_map = attention_map[None]
        attention_map = attention_map / attention_map.max()

        if resize is not None and attention_map.shape[-2:] != (resize, resize):
            attention_map = F.interpolate(attention_map[None], (resize, resize), mode="bicubic")[0].clamp(0, 1)

        return attention_map

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

    def __init__(self, model: StableDiffusionPipeline, source_prompt: str, target_prompt: str, inv_res: Optional[Dict[str, Any]]=None, **kwargs) -> None:
        """Initiates a new prompt-to-prompt controller

        Args:
            model (StableDiffusionPipeline): diffusion model
            source_prompt (str): source prompt for inversion
            target_prompt (str): target prompt for editing
            inv_res (Optional[Dict[str, Any]], optional): Result from inversion. Defaults to None.
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

    def copy(self, **kwargs) -> "PromptToPromptController":
        # for edict
        return PromptToPromptController(self.model, self.source_prompt, self.target_prompt, **self.ptp_cfg)
    

class PromptToPromptControllerAttentionStore(PromptToPromptControllerBase):
    """Prompt-to-prompt controller to retrieve attention maps during diffusion.
    Used by NS-LPIPS to compute object mask
    """

    def __init__(self, model: StableDiffusionPipeline, max_size=32) -> None:
        """Initiates a new prompt-to-prompt controller

        Args:
            model (StableDiffusionPipeline): diffusion model
        """

        # use ptp's attention store
        super().__init__(model, ptp.AttentionStore(max_size=max_size))


class PromptToPromptEditor(ControllerBasedEditor):
    """Prompt-to-prompt editor using a prompt-to-prompt controller for editing
    """

    def __init__(self, inverter: DiffusionInversion, no_source_backward: bool=False, dft_cfg: Dict[Any, str]=None, **kwargs) -> None:
        super().__init__(inverter, no_source_backward, dft_cfg, **kwargs)

    def make_controller(self, image: torch.Tensor, source_prompt: str, target_prompt: str, **kwargs) -> PromptToPromptController:
        return PromptToPromptController(model=self.inverter.model, source_prompt=source_prompt, target_prompt=target_prompt, **kwargs)
        # return PtpController(model=self.inverter.model, source_prompt=source_prompt, target_prompt=target_prompt, **kwargs)


