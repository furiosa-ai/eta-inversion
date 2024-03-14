import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pprint import pprint
import unittest

import numpy as np
import torch
import yaml
from dataset import DatasetBase, load_dataset
from typing import Callable, List, Dict


class TestData(unittest.TestCase):
    """Unit test for datasets. Check if dataset samples and image means match with the tested version.
    """

    # relative dataset indices to test
    test_data_ind = [0, 0.2, 0.5, 1]

    # datasets to test
    datasets = [
        "imagenetr-fake-ti2i",
        "imagenetr-ti2i",
        # "ptp20",
        "pie",
        # "imagenhub",
    ]

    # which keys are required in a dataset item
    required_keys = set([
        "name",
        "image",
        "image_file",
        "source_prompt",
        "target_prompt",
        "mask",
        "edit",
    ])

    # target image means for test dataset
    test_data = {
        'imagenetr-fake-ti2i': {'sample_hashes': {0: 97.35779190063477,
                                           0.2: 89.63984680175781,
                                           0.5: 144.3900286356608,
                                           1: 82.19835535685222}},
        'imagenetr-ti2i': {'sample_hashes': {0: 242.55033365885416,
                                            0.2: 194.51161193847656,
                                            0.5: 153.046511332194,
                                            1: 149.75695292154947}},
        'imagenhub': {'sample_hashes': {0: 137.00004069010416,
                                        0.2: 104.3125597635905,
                                        0.5: 144.51441701253256,
                                        1: 75.59309260050456}},
        'pie': {'sample_hashes': {0: 151.33504486083984,
                                0.2: 100.14775848388672,
                                0.5: 158.48064168294272,
                                1: 108.8532829284668}},
        # 'ptp20': {'sample_hashes': {0: 106.11181640625,
        #                             0.2: 113.71158854166667,
        #                             0.5: 126.65281041463216,
        #                             1: 118.84031295776367}}
    }

    @classmethod
    def rel_to_abs_ind(cls, ind: List[float], l: int) -> List[int]:
        """Converts relative indices to absolute indices"""
        return [round(r * (l - 1)) for r in ind]

    @classmethod
    def get_data_sample_hashes(cls, dataset: DatasetBase) -> Dict[float, float]:
        """Get target means for a dataset

        Args:
            dataset (DatasetBase): Dataset to retrieve target means for

        Returns:
            Dict[float, float]: Dict mapping relative indices to target means
        """

        samples = [dataset[i] for i in cls.rel_to_abs_ind(cls.test_data_ind, len(dataset))]
        hashes = [np.mean(s["image"]).item() for s in samples]
        hashes = dict(zip(cls.test_data_ind, hashes))
        return hashes

    @classmethod
    def generate_test_data(cls):
        """Generate target groundtruth data to test against.
        """

        out = {}
        for dataset_name in cls.datasets:
            dataset = load_dataset(dataset_name)
            img_means = cls.get_data_sample_hashes(dataset)

            out[dataset_name] = {
                "sample_hashes": img_means
            }

        with open("test_data.yaml", "w") as f:
            yaml.dump(out, f)

        pprint(out, depth=4)

    @classmethod
    def setUpClass(cls) -> None:
        """Caches all dataset to test
        """
        cls.datasets_cached = {}
        
        for dataset in cls.datasets:
            try:
                d = load_dataset(dataset)
            except Exception as e:
                d = None
                print(e)

            if d is not None:
                cls.datasets_cached[dataset] = d

    @classmethod
    def generate_test_functions(cls):
        """Automatically generate all test functions for datasets
        """

        for dataset_name in cls.datasets:
            # test if necessary keys exist
            setattr(cls, f"test_{dataset_name}_data_keys", TestData.template_test_data_keys(dataset_name))
            # test if target image mean matches
            setattr(cls, f"test_{dataset_name}_sample_hashes", TestData.template_test_sample_hashes(dataset_name))

    @staticmethod
    def template_test_data_keys(dataset_name: str) -> Callable:
        """Generate test function which checks required dataset keys.

        Args:
            dataset_name (str): Dataset name

        Returns:
            Callable: Test function
        """

        def test_func(self):
            sample_keys = list(self.datasets_cached[dataset_name][0].keys())
            missing_keys = [k for k in self.required_keys if k not in sample_keys]

            self.assertEqual(len(missing_keys), 0, f"Missing keys: {missing_keys}")

        return test_func
    
    @staticmethod
    def template_test_sample_hashes(dataset_name: str)  -> Callable:
        """Generate test function which checks image means.

        Args:
            dataset_name (str): Dataset name

        Returns:
            Callable: Test function
        """

        def test_func(self):
            hashes_cur = self.get_data_sample_hashes(self.datasets_cached[dataset_name])
            hashes_target = self.test_data[dataset_name]["sample_hashes"]

            hashes_cur = [hashes_cur[i] for i in self.test_data_ind]
            hashes_target = [hashes_target[i] for i in self.test_data_ind]

            self.assertListEqual(hashes_cur, hashes_target)

        return test_func


if __name__ == '__main__':
    with torch.no_grad():
        if len(sys.argv) >= 2 and sys.argv[1] == "gen":
            # generate target data
            TestData.generate_test_data()
        else:
            # generate test functions
            TestData.generate_test_functions()
            unittest.main()
