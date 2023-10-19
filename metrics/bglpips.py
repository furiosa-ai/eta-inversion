

import lpips
from metrics.base import SimpleMetric
from modules import load_diffusion_model
import numpy as np
import cv2
from pathlib import PosixPath, Path
import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from numpy import ndarray
from typing import Optional, Tuple


class _BGLPIPSLoss:
    """BG-LPIPS metric. Computing LPIPS on background only (where editing is not desired) using a user-provided fg-bg mask.
    """

    def __init__(self, device: Optional[str]=None, mask_save_path: Optional[PosixPath]=None) -> None:
        """Initializes a new BG-LPIPS metric.

        Args:
            device (Optional[str], optional): Device to use. Defaults to None.
            mask_save_path (Optional[PosixPath], optional): Optionally dumps masks to disk for each __call__ (for debugging). Defaults to None.
        """

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

        # source_image 0-1 [1, 3, 512, 512]
        # target_image 0-1 [1, 3, 512, 512]
        # bg_mask  0-1 [1, 512, 512]
        bg_mask = bg_mask.to(self.device)[:, None]  # Introduce channel dimension
        # mask out foreground
        src_bg, tgt_bg = source_image * bg_mask, target_image * bg_mask
        return self.loss_fn_alex(src_bg, tgt_bg)

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

    def __call__(self, source_image: torch.Tensor, target_image: torch.Tensor, prompt: str, mask: ndarray) -> torch.Tensor:
        """Compute the BG-LPIPS

        Args:
            source_image (torch.Tensor): Groundtruth input image (0-1 range).
            target_image (torch.Tensor): Output image (0-1 range)
            prompt (str): Prompt used for editing (unused)
            mask (ndarray): Mask where 1 is foreground and 0 is background 

        Returns:
            torch.Tensor: Loss value
        """

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        assert mask.dtype == torch.float32

        if mask.ndim == 2:
            # Introduce batch dimension
            mask = mask[None]

        # foreground to background mask
        bg_mask = 1 - mask
        loss = self.crit(source_image, target_image, bg_mask)

        if self.mask_save_path is not None:
            self.save_mask(self.mask_save_path / (prompt + ".png"), bg_mask, fg=True)

        return loss


class BGLPIPS(SimpleMetric):
    """BG-LPIPS metric. Lower means better. Computing LPIPS on background only (where editing is not desired) using a user-provided fg-bg mask.
    """

    def __init__(self, input_range: Tuple[int, int]=(-1, 1), device: Optional[str]=None, mask_save_path: None=None) -> None:
        """BG-LPIPS metric. Computing LPIPS on background only (where editing is not desired) using a user-provided fg-bg mask.

        Args:
            input_range (Tuple[int, int], optional): Input range for image tensors needed for normalization. Defaults to (-1, 1).
            device (Optional[str], optional): Device to compute the metric on. Defaults to None.
            mask_save_path (None, optional): _description_. Defaults to None.
        """

        super().__init__(input_range, device)

        mask_save_path = Path("result/mask")

        # find a free path to dump masks to
        for i in range(999999):
            if not (mask_save_path / str(i)).exists():
                mask_save_path /= str(i)
                mask_save_path.mkdir(parents=True, exist_ok=True)
                break

        self.crit = _BGLPIPSLoss(self.device, mask_save_path=mask_save_path)

    def __repr__(self) -> str:
        return "bglpips"

    def forward(self, source_image: torch.Tensor, target_image: torch.Tensor, prompt: str, mask: Optional[str]=None) -> torch.Tensor:
        """Compute the BG-LPIPS

        Args:
            source_image (torch.Tensor): Groundtruth input image.
            target_image (torch.Tensor): Output image
            prompt (str):  Prompt used for editing (unused)
            mask (Optional[str], optional): Mask where 1 is foreground and 0 is background. If None, None will be returned. Defaults to None.

        Returns:
            torch.Tensor: BG-LPIPS value or None in case of failure
        """
        if mask is not None:
            source_image = self._normalize(source_image)
            target_image = self._normalize(target_image)

            source_image = source_image * 2 - 1  # [0, 1] -> [-1, 1]
            target_image = target_image * 2 - 1  # [0, 1] -> [-1, 1]

            return self.crit(source_image, target_image, prompt, mask)
        else:
            return None
