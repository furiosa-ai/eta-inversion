
import torch
from .base import SimpleMetric
from typing import Optional, Tuple


class MSEMetric(SimpleMetric):
    def __init__(self, input_range=(-1, 1), device: Optional[str]=None) -> None:
        super().__init__(input_range, device)

        self.crit = torch.nn.MSELoss()

    def forward(self, pred, target):
        pred = self._normalize(pred)
        target = self._normalize(target)

        return self.crit(pred, target)

    def __repr__(self) -> str:
        return "mse"
    

class PSNRMetric(SimpleMetric):
    def __init__(self, input_range=(-1, 1), device: Optional[str]=None) -> None:
        super().__init__(input_range, device)

        self.mse_crit = torch.nn.MSELoss()

    def forward(self, pred, target):
        pred = self._normalize(pred)
        target = self._normalize(target)

        mse = self.mse_crit(pred, target)
        return 10 * torch.log10(1 / mse)

    def __repr__(self) -> str:
        return "psnr"


class LPIPSMetric(SimpleMetric):
    """LPIPS metric. Lower means better. 
    """

    def __init__(self, input_range: Tuple[int, int]=(-1, 1), device: Optional[str]=None) -> None:
        super().__init__(input_range, device)

        self.crit_lpips = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # normalize images to 0-1
        pred = self._normalize(pred)
        target = self._normalize(target)

        if self.crit_lpips is None:
            import lpips
            self.crit_lpips = lpips.LPIPS(net='alex').to(pred.device)

        # lpips implementation needs -1 to 1 range
        pred = pred * 2 - 1  # [0, 1] -> [-1, 1]
        target = target * 2 - 1  # [0, 1] -> [-1, 1]
        return self.crit_lpips(pred, target)

    def __repr__(self) -> str:
        return "lpips"
