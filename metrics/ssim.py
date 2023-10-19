
from .base import SimpleMetric
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from typing import Tuple, Optional


class SSIM(SimpleMetric):
    "Structural Similarity Index Measure (SSIM) metric. Ranges from 1 (best) to 0 (worst)."

    def __init__(self, input_range: Tuple[int, int]=(-1, 1), device: Optional[str]=None) -> None:
        super().__init__(input_range, device)

        # use torchmetrics
        self.crit = StructuralSimilarityIndexMeasure().to(self.device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute metric value for a single example.

        Args:
            pred (torch.Tensor): Output image
            target (torch.Tensor): Groundtruth image

        Returns:
            torch.Tensor: Metric value
        """

        pred, target = self._normalize(pred), self._normalize(target)

        return self.crit(pred, target)

    def __repr__(self) -> str:
        return "ssim"
