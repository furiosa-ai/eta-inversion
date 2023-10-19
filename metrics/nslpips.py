

import lpips
from metrics.base import SimpleMetric
from modules import load_diffusion_model
import numpy as np
import cv2
from pathlib import PosixPath, Path
import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from typing import Optional, Tuple

from modules import DiffusionInversion
from modules.editing.ptp_editor import PromptToPromptControllerAttentionStore


class _NSLPIPSLoss:
    """NS-LPIPS metric. Computing LPIPS on background only (where editing is not desired). 
    The fg-bg mask is obtained using Stable Diffusion's attention map of the word to edit in the prompt.
    Refer to https://github.com/sen-mao/StyleDiffusion/blob/master/eval_metrics/ns_lpips.py
    """

    def __init__(self, ldm_stable: StableDiffusionPipeline, mask_save_path: Optional[PosixPath]=None) -> None:
        """Initializes a new NS-LPIPS metric.

        Args:
            ldm_stable (StableDiffusionPipeline): Stable diffusion model to obtain attention masks for foreground-background mask
            mask_save_path (Optional[PosixPath], optional): Optionally dumps masks to disk for each __call__ (for debugging). Defaults to None.
        """

        device = ldm_stable.device

        self.inverter = DiffusionInversion(load_diffusion_model()[0], scheduler="ddim", num_inference_steps=50, guidance_scale_fwd=1)
        self.loss_fn_alex = lpips.LPIPS(net='alex').to(device)  # best forward scores
        self.device = device
        self.mask_save_path = Path(mask_save_path) if mask_save_path is not None else None

        if self.mask_save_path is not None:
            self.mask_save_path.mkdir(parents=True, exist_ok=True)

    def crit(self, source_image: torch.Tensor, target_image: torch.Tensor, bg_mask: torch.Tensor) -> torch.Tensor:
        """Metric criterion

        Args:
            source_image (torch.Tensor): Groundtruth input image (0-1 range).
            target_image (torch.Tensor): Output image (0-1 range)
            bg_mask (torch.Tensor): Mask where 1 is background and 0 is foreground (0-1 range)

        Returns:
            torch.Tensor: Loss
        """

        bg_mask = bg_mask[:, None]
        src_bg, tgt_bg = source_image * bg_mask, target_image * bg_mask
        return self.loss_fn_alex(src_bg, tgt_bg)

    def get_attention_map(self, image: torch.Tensor, prompt: str, word: str) -> torch.Tensor:
        """Retrieves SD UNets' attention map by diffusion inversion. Uses prompt-to-prompt attention store mechanism.

        Args:
            image (torch.Tensor): Input image for Stable Diffusion
            prompt (str): Input prompt for Stable Diffusion
            word (str): Word for which to extract attention maps. Must be in the prompt

        Returns:
            torch.Tensor: Attention map resized to 512x512 and normalized to 0-1. Shape is [1, 512, 512].
        """

        # use prompt-to-prompt's attention map store to cache attention map during diffusion
        controller = PromptToPromptControllerAttentionStore(self.inverter.model)

        # run diffusion inversion
        with self.inverter.use_controller(controller):
            _ = self.inverter.invert(image, prompt=prompt, guidance_scale_fwd=1)

        # retrieve stored attention maps
        attention_map = controller.get_attention_map(prompt, word, resize=512)
        return attention_map

    def get_bg_mask(self, image: torch.Tensor, prompt: str, word: str) -> torch.Tensor:
        """Gets attention map and returns its corresponding background mask

        Args:
            image (torch.Tensor): Input image for Stable Diffusion
            prompt (str): Input prompt for Stable Diffusion
            word (str): Word for which to extract attention maps. Must be in the prompt

        Returns:
            torch.Tensor: Background mask. Shape is [1, 512, 512].
        """

        attention_map = self.get_attention_map(image, prompt, word)

        # invert
        return 1 - attention_map

    def save_mask(self, filename: PosixPath, mask: torch.Tensor, fg: bool=False) -> None:
        """Helper function to save foreground mask

        Args:
            filename (PosixPath): Path to save image to
            mask (torch.Tensor): Mask tensor
            fg (bool, optional): If True mask will be inverted before saved. Defaults to False.
        """
        if fg:
            mask = 1 - mask

        mask = (255 * mask.cpu().numpy()).astype(np.uint8)[0]
        cv2.imwrite(str(filename), mask)

    def __call__(self, source_image: torch.Tensor, target_image: torch.Tensor, prompt: str, word: str) -> torch.Tensor:
        """Compute the NS-LPIPS

        Args:
            source_image (torch.Tensor): Groundtruth input image (0-1 range).
            target_image (torch.Tensor): Output image (0-1 range)
            prompt (str): Prompt used for editing
            word (str): Changed word in target prompt

        Returns:
            torch.Tensor: Loss value
        """

        bg_mask = self.get_bg_mask(source_image, prompt, word)
        loss = self.crit(source_image, target_image, bg_mask)

        if self.mask_save_path is not None:
            self.save_mask(self.mask_save_path / (prompt + ".png"), bg_mask, fg=True)

        return loss


class NSLPIPS(SimpleMetric):
    """NS-LPIPS metric. Lower means better. Computing LPIPS on background only (where editing is not desired). 
    The fg-bg mask is obtained using Stable Diffusion's attention map of the word to edit in the prompt.
    """

    def __init__(self, input_range: Tuple[int, int]=(-1, 1), device: Optional[str]=None, mask_save_path: Optional[str]=None) -> None:
        """Initializes a new NS-LPIPS metric.

        Args:
            input_range (Tuple[int, int], optional): Input range for image tensors needed for normalization. Defaults to (-1, 1).
            device (Optional[str], optional): Device to compute the metric on. Defaults to None.
            mask_save_path (Optional[PosixPath], optional): Optionally dumps masks to disk for each __call__ (for debugging). Defaults to None.
        """

        super().__init__(input_range, device)
    
        # load stable diffusion
        model, _ = load_diffusion_model(device=self.device)

        mask_save_path = Path("result/mask")

        # find a free path to dump masks to
        for i in range(999999):
            if not (mask_save_path / str(i)).exists():
                mask_save_path /= str(i)
                mask_save_path.mkdir(parents=True, exist_ok=True)
                break

        self.crit = _NSLPIPSLoss(model, mask_save_path=mask_save_path)

    def __repr__(self) -> str:
        return "nslpips"

    def forward(self, source_image: torch.Tensor, target_image: torch.Tensor, prompt: str, word: str) -> torch.Tensor:
        """Compute the NS-LPIPS

        Args:
            source_image (torch.Tensor): Groundtruth input image
            target_image (torch.Tensor): Output image
            prompt (str): Prompt used for editing
            word (str): Word to edit

        Returns:
            torch.Tensor: Loss value
        """

        assert isinstance(source_image, torch.Tensor) and isinstance(target_image, torch.Tensor)
        if word is not None:
            source_image = self._normalize(source_image)
            target_image = self._normalize(target_image)

            source_image = source_image * 2 - 1  # [0, 1] -> [-1, 1]
            target_image = target_image * 2 - 1  # [0, 1] -> [-1, 1]

            loss = self.crit(source_image, target_image, prompt, word)
            return loss
        else:
            return None
