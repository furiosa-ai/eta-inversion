import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
import hashlib

import torch
import cv2
import numpy as np
import json
import yaml
from typing import Callable, Dict, Any, Tuple, List, Union

from eval import main as eval_main
from compute_metrics import main as compute_metrics_main


class TestEval(unittest.TestCase):
    """Unit test for evaluation and metric computation code. 
    Checks if computed metrics over a dataset match with the target.
    Target data is also computed using this class beforehand.
    """

    test_data = None

    # test metrics
    metrics = [
        "clip",
        "clip_acc",
        "blipclip",
        "blipclip_acc",
        "dinovitstruct",
        "dinovitstruct_v2",
        "clip_pix2pix",
        "lpips",
        "nslpips",
        "bglpips",
        "ssim",
    ]

    device = "cuda"
    eval_name = "eval_test"

    # output directories and files
    eval_cfg = f"test/data/{eval_name}.yaml"
    eval_path = f"result/{eval_name}"

    # were gt result is stored
    eval_res_file = f"test/data/{eval_name}_res.yaml"

    @classmethod
    def get_test_data_by_exp(cls, mode: str) -> Dict[str, Any]:
        """Load target data for testing

        Args:
            mode (str): Experiment to load

        Returns:
            Dict[str, Any]: Eval data
        """

        out = {}

        for file, data in cls.test_data[mode].items():
            rel_path = Path(file).relative_to(cls.eval_path)
            exp = str(rel_path.parents[len(rel_path.parents) - 2] if len(rel_path.parents) > 1 else "")

            if exp == "":
                continue

            if exp not in out:
                out[exp] = {}

            out[exp][file] = data

        return out

    def assert_file_hashes(self, filenames: List[str]) -> None:
        """Assert output evaluation file matches stored target evaluation file hash

        Args:
            filenames (List[str]): Files to check
        """

        for output_file in filenames:
            with self.subTest():
                with self.subTest():
                    self.assertTrue(Path(output_file).exists(), f"{output_file} not found")

                output_hash = self.get_file_cmp_hash(output_file)
                # target_hash = self.get_file_cmp_hash(gt_file) # self.test_data["eval_files"][gt_file]["value"]
                target_hash = self.test_data["eval_files"][output_file]["value"]

                with self.subTest():
                    self.assertAlmostEqual(output_hash, target_hash, msg=f"{output_file} not matching", places=None if "_nti_" not in self.id() else 3)

    def assert_metric_files(self, filenames: List[str]) -> None:
        """Assert output metric file matches stored target metric file hash

        Args:
            filenames (List[str]): Files to check
        """

        for output_file in filenames:
            with self.subTest():
                with self.subTest():
                    self.assertTrue(Path(output_file).exists(), f"{output_file} not found")

                with open(output_file, "r") as fp:
                    metrics_output = yaml.safe_load(fp)

                # with open(gt_file, "r") as fp:
                #     metrics_target = yaml.safe_load(fp)

                metrics_target = self.test_data["metrics"][output_file]

                with self.subTest():
                    self.assertEqual(metrics_output["name"], metrics_target["name"])

                with self.subTest():
                    if np.isnan(metrics_target["mean"]):
                        print(f"Metric missing for {output_file}")
                    else:
                        self.assertAlmostEqual(metrics_output["mean"], metrics_target["mean"], places=None if "_nti_" not in self.id() else 3)

                # test all images separately?

    @classmethod
    def get_file_cmp_hash(cls, filename: str) -> Union[int, float]:
        """Computes hash of mean over file for comparison

        Args:
            filename (str): File to compute hash for

        Returns:
            Union[int, float]: Compare value
        """

        filename = Path(filename)

        if filename.suffix == ".yaml":
            with open(filename, "r") as f:
                dic = yaml.safe_load(f)
            return int(hashlib.sha1(json.dumps(dic, sort_keys=True).encode()).hexdigest(), 16)
        elif filename.suffix == ".png":
            image = cv2.cvtColor(cv2.imread(str(filename)), cv2.COLOR_BGR2RGB)
            return np.mean(image).item()
        else:
            raise Exception(f"Unknown suffix for {filename}")

    @classmethod
    def list_eval_files(cls) -> List[str]:
        """Get all stored target eval files

        Returns:
            List[str]: Eval file paths
        """

        return [str(p) for p in sorted(Path(cls.eval_path).rglob("**/*.*")) if p.name == "cfg.yaml" or p.suffix == ".png"]
    
    @classmethod
    def list_metric_files(cls) -> List[str]:
        """Get all stored target metric files

        Returns:
            List[str]: Metric file paths
        """

        return [str(p) for p in sorted(Path(cls.eval_path).rglob("**/*.*")) if p.parent.name == "metrics"]

    @classmethod
    def run_eval_compute(cls) -> None:
        """Generate images and compute metrics. Both used for target data and output data.
        """

        eval_main(cfg=[cls.eval_cfg], device=None, no_proc=False)
        compute_metrics_main(cfg=[cls.eval_cfg], metric=cls.metrics)

    @classmethod
    def generate_test_data(cls) -> None:
        """Generates target test data. Should be run beforehand and not rerun afterwards.
        """

        # run eval and compute metrics to generate groundtruth
        cls.run_eval_compute()

        metric_files = cls.list_metric_files()
        metrics = {}
        for metric_file in metric_files:
            with open(metric_file, "r") as f:
                metrics[metric_file] = yaml.safe_load(f)

        out = {
            "eval_files": {name: {"value": cls.get_file_cmp_hash(name)} for name in cls.list_eval_files()},
            "metrics": metrics,
        }

        with open(cls.eval_gt_res_file, "w") as f:
            yaml.dump(out, f)

        print(json.dumps(out, indent=4))

    @classmethod
    def generate_test_functions(cls) -> None:
        """Automatically generate all test functions for testing.
        Checks if eval output files and metric files matches with target.
        """

        # run eval and compute metrics to compare with groundtruth
        cls.run_eval_compute()
        cls.load_test_gt_data()

        test_gt_data_by_exp = cls.get_test_data_by_exp("eval_files")
        for exp_name, test_data in test_gt_data_by_exp.items():
            setattr(cls, f"test_{exp_name}_eval_equal", TestEval.template_test_exp_eval_equal(test_data.keys()))

        test_gt_data_by_exp = cls.get_test_data_by_exp("metrics")
        for exp_name, test_data in test_gt_data_by_exp.items():
            setattr(cls, f"test_{exp_name}_metrics_equal", TestEval.template_test_exp_metrics_equal(test_data.keys()))

    @classmethod
    def load_test_gt_data(cls) -> None:
        """Loads taget data
        """

        with open(cls.eval_res_file, "r") as f:
            cls.test_data = yaml.safe_load(f)

    @classmethod
    def setUpClass(cls) -> None:
        pass

    def test_config_equal(self) -> None:
        """Test for global config file. Hash must match target.
        """

        cfg_file = f"result/{self.eval_name}/cfg.yaml"
        self.assert_file_hashes([cfg_file])

    @staticmethod
    def template_test_exp_eval_equal(filenames: List[str]) -> Callable:
        """Generate test function which checks if output eval file matches target.

        Args:
            filenames (List[str]): Files to test

        Returns:
            Callable: Test function
        """

        def test_func(self):
            self.assert_file_hashes(filenames)
        return test_func
    
    @staticmethod
    def template_test_exp_metrics_equal(filenames: List[str]) -> Callable:
        """Generate test function which checks if output metric file matches target.

        Args:
            filenames (List[str]): Files to test

        Returns:
            Callable: Test function
        """

        def test_func(self):
            self.assert_metric_files(filenames)
        return test_func


if __name__ == '__main__':
    with torch.no_grad():
        if len(sys.argv) >= 2 and sys.argv[1] == "gen":
            # generate target data
            TestEval.generate_test_data()
        else:
            # generate test functions
            TestEval.generate_test_functions()
            unittest.main()
