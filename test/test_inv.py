import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.debug_utils import enable_deterministic
enable_deterministic()

import unittest

import torch
import cv2
import yaml
import json
from diffusers import StableDiffusionPipeline
from modules import load_inverter
from modules import StablePreprocess, StablePostProc

from typing import Callable, Dict, Any


class TestInv(unittest.TestCase):
    """Unit test for inversion methods. Check if inverted image mean matches the target mean.
    """
    
    # Test image and prompt for inversion
    test_img = "test/data/gnochi_mirror_sq.png"
    test_prompt = "a cat sitting next to a mirror"
    test_crit = "image"  # latent

    # inversion methods to test
    inversion_methods = [
        dict(type="diffinv", scheduler="ddim", num_inference_steps=50),
        dict(type="nti", scheduler="ddim", num_inference_steps=50),
        dict(type="npi", scheduler="ddim", num_inference_steps=50),
        dict(type="proxnpi", scheduler="ddim", num_inference_steps=50),
        dict(type="edict", scheduler="ddim", num_inference_steps=50),
        dict(type="ddpminv", scheduler="ddpm", num_inference_steps=50),
        dict(type="dirinv", scheduler="ddim", num_inference_steps=50),
        dict(type="etainv", scheduler="ddim", num_inference_steps=50),
    ]

    # target means
    test_data = {
        "diffinv_ddim_50": -0.003393499180674553,
        "nti_ddim_50": -0.005135257262736559,
        "npi_ddim_50": -0.008206297643482685,
        "proxnpi_ddim_50": -0.008206297643482685,
        "edict_ddim_50": -0.007414111401885748,
        "ddpminv_ddpm_50": 0.002014702884480357,
        "dirinv_ddim_50": -0.0074142711237072945,
        "etainv_ddim_50":  -0.0074142711237072945,
    }

    @classmethod
    def setUp(cls):
        """Prepare test case
        """

        # reset seed
        enable_deterministic()
    
    @classmethod
    def setUpClass(cls):
        """Load Stable Diffusion model and prepare inversion methods
        """

        # enable_deterministic()  # does not work with nti
        cls.inv_cfgs = {}

        for i, inversion_method in enumerate(cls.inversion_methods):
            # name = f"{i}_{cfg['type']}_{cfg['scheduler']}"
            name = "_".join([str(v) for v in inversion_method.values()])
            cls.inv_cfgs[name] = inversion_method

        cls.save_image_path = Path("result/test")
        cls.device = "cuda"

        cls.model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(cls.device)
        cls.preproc = StablePreprocess(cls.device, size=512, **{"center_crop": True, "return_np": False})
        cls.postproc = StablePostProc()

    @classmethod
    def invert_helper(cls, inv_config: Dict[str, Any], image: str, prompt: str) -> Dict[str, Any]:
        """Invert image and return result.

        Args:
            inv_config (Dict[str, Any]): Config for inverter
            image (str): Image to invert
            prompt (str): Prompt for inversion

        Returns:
            Dict[str, Any]: Inversion result
        """

        image = cls.preproc(image)
        inverter = load_inverter(model=cls.model, **inv_config)
        inv_res = inverter.invert_sample(image, prompt)
        return inv_res

    @classmethod
    def generate_test_data(cls):
        """Generate and print target data.
        """

        cls.setUpClass()

        out = {}
        for test_name, inv_cfg in cls.inv_cfgs.items():
            # invert image and log mean
            inv_res = cls.invert_helper(inv_cfg, image=cls.test_img, prompt=cls.test_prompt)

            if inv_res is None:
                print(f"{test_name} not supported")
            else:
                target_mean = torch.mean(inv_res[cls.test_crit]).item()
                out[test_name] = target_mean

        with open("test_inv.yaml", "w") as f:
            yaml.dump(out, f)

        print(json.dumps(out, indent=4))

    @classmethod
    def generate_test_functions(cls) -> None:
        """Automatically generate all test functions for inversion.
        Checks if inversion methods yield correct output image mean.
        """

        cls.setUpClass()

        for test_name, inv_cfg in cls.inv_cfgs.items():
            # generate tests for each inversion method
            setattr(cls, f"test_{test_name}_equal", TestInv.template_test_inv(test_name, inv_cfg, cls.test_img, cls.test_prompt))

    def save_img_res(self, path: str, image: torch.Tensor) -> None:
        """Saves image result for debugging

        Args:
            path (str): Path to save image
            image (torch.Tensor): image tensor
        """

        img_inv = self.postproc(image)
        cv2.imwrite(str(path), cv2.cvtColor(img_inv, cv2.COLOR_RGB2BGR))

    def assert_mean_almost_equal(self, tensor, target_mean, places=None):
        self.assertAlmostEqual(torch.mean(tensor).item(), target_mean, places=places)

    @staticmethod
    def template_test_inv(test_name: str, inv_cfg: Dict[str, Any], image: str, prompt: str) -> Callable:
        """Generate test function which checks output of inversion matches target mean.

        Args:
            test_name (str): Name of the test
            inv_cfg (Dict[str, Any]): Inversion config to test
            image (str): Image path to test
            prompt (str): Image prompt

        Returns:
            Callable: Test function
        """

        def test_func(self):
            inv_res = self.invert_helper(inv_cfg, image, prompt)
            img_inv = inv_res[self.test_crit]

            if self.save_image_path is not None:
                self.save_image_path.mkdir(parents=True, exist_ok=True)
                self.save_img_res(self.save_image_path / (test_name + ".png"), img_inv)

            mean_target = self.test_data[test_name]

            self.assert_mean_almost_equal(img_inv, mean_target, places=5 if inv_cfg["type"] == "nti" else None)  # nti is non-deterministic
            # self.assert_mean_almost_equal(edit_res["latent"], target_mean)

        return test_func


if __name__ == '__main__':
    with torch.no_grad():
        if len(sys.argv) >= 2 and sys.argv[1] == "gen":
            # generate target data
            TestInv.generate_test_data()
        else:
            # generate test functions
            TestInv.generate_test_functions()
            unittest.main()
