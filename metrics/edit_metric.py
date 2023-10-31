import torch
import numpy as np
from functools import partial
from .clip_similarity import CLIPSimilarity
from .clip_similarity import CLIPAccuracy
from .dino_vit_structure import DinoVitStructure
from .metrics import LPIPSMetric
from .nslpips import NSLPIPS
from .bglpips import BGLPIPS
from .ssim import SSIM
from .msssim import MSSSIM
from .base import SimpleMetric
from typing import Dict, List, Optional, Tuple, Union


class EditMetric(SimpleMetric):
    """Class for managing various editing metrics and providing a unified interface.
    """

    def __init__(self, name: str, input_range: Tuple[int, int]=(-1, 1), device: Optional[str]=None, **kwargs) -> None:
        """Initializes a new metric object.

        Args:
            name (str): Name of the metric to create. Must be a metric from get_available_metrics().
            input_range (Tuple[int, int], optional): Input range for image tensors needed for normalization. Defaults to (-1, 1).
            device (Optional[str], optional): Device to compute the metric on. Defaults to None.
        """

        super().__init__(input_range, device)

        self.metric_name = name
        self.metric = {
            "clip_text_img": partial(CLIPSimilarity, metric="text_img"),
            "clip_img_img": partial(CLIPSimilarity, metric="img_img"),
            "clip_text_text": partial(CLIPSimilarity, metric="text_text"),
            "clip_textdir_imgdir": partial(CLIPSimilarity, metric="textdir_imgdir"),
            "clip_text_img_acc": partial(CLIPAccuracy, metric="text_img"),
            "clip_text_text_acc": partial(CLIPAccuracy, metric="text_text"),
            "dinovitstruct": DinoVitStructure,
            "dinovitstruct_v2": partial(DinoVitStructure, vit_model="dinov2_vitb14"),
            "lpips": LPIPSMetric,
            "nslpips": NSLPIPS,
            "bglpips": BGLPIPS,
            "ssim": SSIM,
            "msssim": MSSSIM,
        }[self.metric_name](input_range=input_range, device=device, **kwargs)

    @staticmethod
    def get_available_metrics() -> List[str]:
        """Returns a list of names of all possible metrics.

        Returns:
            List[str]: Metric list
        """
        return [
            "clip_text_img",
            "clip_img_img",
            "clip_text_text",
            "clip_textdir_imgdir",
            "clip_text_img_acc",
            "clip_text_text_acc",
            "dinovitstruct",
            "dinovitstruct_v2",
            "lpips",
            "nslpips",
            "bglpips",
            "ssim",
            "msssim",
        ]

    def update(self, source_image: torch.Tensor, edit_image: torch.Tensor, source_prompt: str, target_prompt: str, edit_word: str, mask: Optional[torch.Tensor]=None) -> float:
        """Compute metric for the given example and add record its result. 
        After adding all examples via this method use compute() to compute the final metric average over all examples.

        Args:
            source_image (torch.Tensor): Groundtruth input image.
            edit_image (torch.Tensor): Output image.
            source_prompt (str): Prompt used for inversion.
            target_prompt (str): Prompt used for editing/denoising.
            edit_word (str): Changed word in target prompt
            mask (Optional[torch.Tensor], optional): Mask where 1 is foreground and 0 is background . Defaults to None.

        Returns:
            float: Metric value for the provided example or None in case of failure
        """

        # select required arguments for respective metric
        args = {
            "dinovitstruct": (source_image, edit_image),
            "dinovitstruct_v2": (source_image, edit_image),
            "lpips": (source_image, edit_image),
            "nslpips": (source_image, edit_image, source_prompt, edit_word),
            "bglpips": (source_image, edit_image, source_prompt, mask),
            "ssim": (edit_image, source_image), 
            "msssim": (edit_image, source_image), 
        }.get(self.metric_name, dict(
            source_image=source_image, target_image=edit_image, 
            source_prompt=source_prompt, target_prompt=target_prompt),
        )

        # compute loss and convert to float
        loss = self.metric.update(*args) if isinstance(args, tuple) else self.metric.update(**args)

        if isinstance(loss, (torch.Tensor,)):
            loss = loss.item()
        elif isinstance(loss, (np.ndarray)):
            loss = loss.astype(float).item()

        assert loss is None or isinstance(loss, float), f"{type(loss)}"
        return loss

    def compute(self) -> Tuple[float, Dict[str, Union[float, List[float]]]]:
        return self.metric.compute()

    def __repr__(self) -> str:
        return self.metric.__repr__()
