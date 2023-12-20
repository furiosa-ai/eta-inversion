import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest

import torch
import cv2
import numpy as np
import json
from typing import Callable, Dict, Any, Union, Optional

from modules import StablePreprocess
from metrics.edit_metric import EditMetric


def _load_mask(file):
    """load image mask"""
    mask = cv2.imread(file, 0)
    assert mask.ndim == 2
    mask = mask.astype(np.float32) / 255
    return mask


class TestMetrics(unittest.TestCase):
    """Unit test for metrics. Check if metric result matches with the target.
    """

    # metrics to test
    metrics = EditMetric.get_available_metrics()

    device = "cuda"

    # image input and target metric values
    test_cat = "cat"
    test_data = {
        "cat": {
            "data": {
                "source_image": "test/data/gnochi_mirror_sq.png",
                "edit_image": "test/data/gnochi_mirror_sq_edit_example.png",
                "source_prompt": "a cat sitting next to a mirror",
                "target_prompt": "a tiger sitting next to a mirror",
                "edit_word": "cat",
                "mask": "test/data/gnochi_mirror_sq_mask.png",
            },
            "metrics": {
                "clip_text_img": 0.32212701439857483,
                "clip_img_img": 0.6910541653633118,
                "clip_text_text": 0.9205214381217957,
                "clip_textdir_imgdir": 0.1089695394039154,
                "clip_text_img_acc": 1.0,
                "clip_text_text_acc": 1.0,
                "dinovitstruct": 0.018216347321867943, 
                "dinovitstruct_v2": 0.003991228528320789, 
                "lpips": 0.24533388018608093, 
                "nslpips": 0.22590836882591248, 
                "bglpips": 0.0347834937274456, 
                "ssim": 0.6813936829566956,
                "msssim": 0.7749947905540466,
                "mse": 0.011490068398416042,
                "psnr": 19.396774291992188,
            }
        }
    }

    @classmethod
    def load_test_data(cls) -> Dict[str, Any]:
        """Load test image data

        Returns:
            Dict[str, Any]: Test data (source image, edit image, mask)
        """
        
        data = cls.test_data[cls.test_cat]["data"]

        data["source_image"] = cls.preproc(data["source_image"])
        data["edit_image"] = cls.preproc(data["edit_image"])
        data["mask"] = _load_mask(data["mask"])

        return data

    @classmethod
    def setUpClass(cls):
        cls.load_test_data()

    @classmethod
    def generate_test_data(cls) -> None:
        """Generate and print target data.
        """

        cls.setUpClass()

        print("Generating test data")
        metrics = {metric: cls.create_metric(metric) for metric in cls.metrics}

        out = {}

        out[cls.test_cat] = {}
        for metric_name, metric in metrics.items():
            loss = cls.metric_helper(metric, **cls.test_data[cls.test_cat]["data"])
            out[cls.test_cat][metric_name] = loss

        print(json.dumps(out, indent=4))   

    @classmethod
    def generate_test_functions(cls):
        """Automatically generate all test functions for metrics.
        """

        for metric in cls.metrics:
            setattr(cls, f"test_{metric}_equal", TestMetrics.template_test_metric_equals(cls.test_cat, metric))         

    @classmethod
    def preproc(cls, image: str) -> torch.Tensor:
        """Load and preprocess image

        Args:
            image (str): Path to image

        Returns:
            torch.Tensor: Metric input
        """
        return StablePreprocess(cls.device, size=512, **{"center_crop": True, "return_np": False})(image)

    @classmethod
    def create_metric(cls, metric: str) -> EditMetric:
        """Load metric from name

        Args:
            metric (str): Metric name

        Returns:
            EditMetric: Metric
        """
        return EditMetric(metric, input_range=(-1, 1), device=cls.device)

    @classmethod
    def metric_helper(cls, metric: Union[str, EditMetric], source_image: torch.Tensor, 
                      edit_image: torch.Tensor, source_prompt: str, target_prompt: str, edit_word: str, 
                      mask: Optional[np.ndarray]=None) -> float:
        """Evaluate metric for a sample and return result value

        Args:
            metric (Union[str, EditMetric]): Metric name or object
            source_image (torch.Tensor): Source input image
            edit_image (torch.Tensor): Edited output image
            source_prompt (str): Source prompt
            target_prompt (str): Target prompt
            edit_word (str): Word changed between source and target
            mask (Optional[np.ndarray], optional): Fg-bg mask (1 is foreground). Defaults to None.

        Returns:
            float: Metric value
        """

        if isinstance(metric, str):
            metric = cls.create_metric(metric)
        
        metric.update(
            source_image=source_image, 
            edit_image=edit_image, 
            source_prompt=source_prompt, 
            target_prompt=target_prompt,
            edit_word=edit_word,
            mask=mask
        )

        return metric.compute()[0]

    def assert_mean_almost_equal(self, x_pred, x_gt, places=None):
        self.assertAlmostEqual(x_pred, x_gt, places=places)

    @staticmethod
    def template_test_metric_equals(cat: str, metric: str) -> Callable:
        """Generate test function which checks output of metric matches target.

        Args:
            cat (str): Category to evaluate
            metric (str): Metric name

        Returns:
            Callable: Test function
        """

        def test_func(self):
            output_loss = self.metric_helper(
                metric,
                **self.test_data[cat]["data"]
            )

            target_loss = self.test_data[cat]["metrics"].get(metric, None)

            self.assertIsNotNone(target_loss, f"No test data found. Output metric is {output_loss}.")
            self.assert_mean_almost_equal(output_loss, target_loss)
        return test_func


if __name__ == "__main__":
    with torch.no_grad():
        if len(sys.argv) >= 2 and sys.argv[1] == "gen":
            # generate target data
            TestMetrics.generate_test_data()
        else:
            # generate test functions
            TestMetrics.generate_test_functions()
            unittest.main()
