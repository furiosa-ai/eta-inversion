import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


class BaseMetric:
    """Base class for all metrics.
    """

    def __init__(self, input_range: Tuple[int, int]=(-1, 1), device: Optional[str]=None) -> None:
        """Initializes a new metric object.

        Args:
            input_range (Tuple[int, int], optional): Input range for image tensors needed for normalization. Defaults to (-1, 1).
            device (Optional[str], optional): Device to compute the metric on. Defaults to None.
        """

        self.input_range = input_range

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes image tensor to 0-1 range.

        Args:
            x (torch.Tensor): Tensor to normalize

        Returns:
            torch.Tensor: Normalized tensor
        """
        if self.input_range is None:
            return x

        return (x - self.input_range[0]) / (self.input_range[1] - self.input_range[0])

    def __call__(self, *args, **kwargs) -> Union[torch.Tensor, None]:
        """Compute metric value for a single example.

        Returns:
            Union[torch.Tensor, None]: Metric value or None in case of failure
        """
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Union[torch.Tensor, None]:
        """Compute metric value for a single example.

        Returns:
            Union[torch.Tensor, None]: Metric value or None in case of failure
        """
        raise NotImplementedError
    
    def update(self, *args, **kwargs) -> Union[float, None]:
        """Compute metric for the given example and add record its result. 
        After adding all examples via this method use compute() to compute the final metric average over all examples.

        Returns:
            Union[float, None]: Metric value for the provided example or None in case of failure
        """
        raise NotImplementedError
    
    def compute(self) -> Tuple[float, Dict[str, Union[float, List[float]]]]:
        """Computes the final averaged metric value for all examples added via update().

        Returns:
            Tuple[float, Dict[str, Union[float, List[float]]]]: A pair where the first value is the metric value (float scalar) 
            and the second value is a dict containing additional results.
        """
        raise NotImplementedError


class SimpleMetric(BaseMetric):
    """Class which implemets some basic methods of BaseMetric
    """

    def __init__(self, input_range: Tuple[int, int]=(-1, 1), device: Optional[str]=None) -> None:
        super().__init__(input_range=input_range, device=device)

        self.losses = []

    def update(self, *args, **kwargs) -> Union[float, None]:
        loss = self.forward(*args, **kwargs)

        if loss is None:
            return None

        # add loss to list
        self.losses.append(loss.detach().cpu().numpy().item())
        return self.losses[-1]

    def compute(self) -> Tuple[float, Dict[str, Union[float, List[float]]]]:
        # final loss as mean
        res = np.mean(self.losses).astype(float).item()
        out = res, {
            "value": res,
            "all": self.losses,
        }
        # reset loss list
        self.losses = []
        return out