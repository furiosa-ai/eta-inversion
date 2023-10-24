import sys
from pathlib import Path
from typing import Any, Callable, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.debug_utils import enable_deterministic
enable_deterministic()

from itertools import product
import unittest

import torch
import cv2
import yaml
import json
from diffusers import StableDiffusionPipeline
from modules import load_editor, load_inverter
from modules import StablePreprocess, StablePostProc


class TestEdit(unittest.TestCase):
    """Unit test for inversion methods. Check if inverted image mean matches the target mean.
    """

    # Test image, source and target prompt for editing
    test_img = "test/data/gnochi_mirror_sq.png"
    test_source_prompt = "a cat sitting next to a mirror"
    test_target_prompt = "a tiger sitting next to a mirror"
    steps = 50

    # additional edit config for ptp
    test_edit_method_configs = { 
        "ptp": {
            "is_replace_controller": True,
            "cross_replace_steps": {"default_": .8},
            "self_replace_steps": .5,
            "blend_words": [["cat"], ["tiger"]], 
            "equilizer_params": {"words": ["tiger"], "values": [2]},
    }}

    test_crit = "image"  # latent

    # inversion methods to test
    inversion_methods = [
        dict(type="diffinv", scheduler="ddim", num_inference_steps=steps),
        dict(type="nti", scheduler="ddim", num_inference_steps=steps),
        dict(type="npi", scheduler="ddim", num_inference_steps=steps),
        dict(type="proxnpi", scheduler="ddim", num_inference_steps=steps),
        dict(type="edict", scheduler="ddim", num_inference_steps=steps),
        dict(type="ddpminv", scheduler="ddpm", num_inference_steps=steps),
    ]

    # edit methods to test
    edit_methods = [
        "simple", 
        "ptp", 
        "masactrl", 
        "pnp",
        "pix2pix_zero",
    ]

    # target means
    # test functions will be named: test_{key}_equal
    test_data = {
        "diffinv_ddim_50__simple": -0.06234968826174736,
        "diffinv_ddim_50__ptp": -0.04683462902903557,
        "diffinv_ddim_50__masactrl": -0.053135693073272705,
        "diffinv_ddim_50__pnp": -0.0030027353204786777,
        "diffinv_ddim_50__pix2pix_zero": -0.12201599776744843,
        "nti_ddim_50__simple": -0.029435917735099792,
        "nti_ddim_50__ptp": -0.046503257006406784,
        "nti_ddim_50__masactrl": -0.05268318951129913,
        "nti_ddim_50__pnp": -0.06137224659323692,
        "nti_ddim_50__pix2pix_zero": -0.11792322993278503,
        "npi_ddim_50__simple": -0.05262122303247452,
        "npi_ddim_50__ptp": -0.04142346978187561,
        "npi_ddim_50__masactrl": -0.053135693073272705,
        "npi_ddim_50__pnp": -0.06164667010307312,
        "npi_ddim_50__pix2pix_zero": -0.12201599776744843,
        "proxnpi_ddim_50__simple": -0.0467924103140831,
        "proxnpi_ddim_50__ptp": -0.041878968477249146,
        "proxnpi_ddim_50__masactrl": -0.03455960750579834,
        "proxnpi_ddim_50__pnp": -0.03140567988157272,
        "proxnpi_ddim_50__pix2pix_zero": -0.023641865700483322,
        "edict_ddim_50__simple": -0.015604550018906593,
        "edict_ddim_50__ptp": -0.02095811814069748,
        "edict_ddim_50__masactrl": -0.012362940236926079,
        "edict_ddim_50__pix2pix_zero": 0.0015177088789641857,
        "ddpminv_ddpm_50__simple": -0.02269146963953972,
        "ddpminv_ddpm_50__ptp": -0.03735125809907913,
        "ddpminv_ddpm_50__masactrl": -0.020472053438425064,
        "ddpminv_ddpm_50__pnp": -0.008612200617790222,
        "ddpminv_ddpm_50__pix2pix_zero": -0.006461526267230511
    }

    # TestEdit.test_diffinv_ddim_50__pix2pix_zero_equal TestEdit.test_ddpminv_ddpm_50__pix2pix_zero_equal

    def setUp(self):
        """Prepare test case
        """

        # reset seed
        enable_deterministic()

    @classmethod
    def setUpClass(cls):
        """Load Stable Diffusion model and prepare inversion and editing methods
        """

        cls.edit_cfgs = {}

        for i, (inversion_method, edit_method) in enumerate(product(cls.inversion_methods, cls.edit_methods)):
            # name = f"{i}_{cfg['type']}_{cfg['scheduler']}"

            if isinstance(edit_method, str):
                edit_method = dict(type=edit_method)
            
            name = "_".join([str(v) for v in inversion_method.values()]) + "__" + "_".join([str(v) for v in edit_method.values()])
            cls.edit_cfgs[name] = {"inversion": inversion_method, "edit": edit_method}

        cls.save_image_path = Path("result/test")
        cls.device = "cuda"

        cls.model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(cls.device)
        cls.preproc = StablePreprocess(cls.device, size=512, **{"center_crop": True, "return_np": False})
        cls.postproc = StablePostProc()

    @classmethod
    def edit_helper(cls, edit_config: Dict[str, Any], image: str, source_prompt: str, 
                    target_prompt: str, edit_add_config: Dict[str, Any]) -> Dict[str, Any]:
        """Invert and edit image and return result.

        Args:
            edit_config (Dict[str, Any]): Config for editor
            image (str): Image to edit
            source_prompt (str): Source prompt (inversion)
            target_prompt (str): Target prompt (editing)
            edit_add_config (Dict[str, Any]): Additional config (ptp)

        Returns:
            Dict[str, Any]: Editing result
        """

        image = cls.preproc(image)
        inverter = load_inverter(model=cls.model, **edit_config["inversion"])
        editor = load_editor(inverter=inverter, **edit_config["edit"])
        edit_res = editor.edit(image, source_prompt, target_prompt, edit_add_config)
        return edit_res
    
    @classmethod
    def generate_test_data(cls):
        """Generate and print target data.
        """

        cls.setUpClass()

        print(list(cls.edit_cfgs.keys()))

        out = {}
        for test_name, edit_cfg in cls.edit_cfgs.items():
            edit_res = cls.edit_helper(edit_cfg, cls.test_img, cls.test_source_prompt, cls.test_target_prompt, cls.test_edit_method_configs.get(edit_cfg["edit"]["type"], None))

            if edit_res is None:
                print(f"{test_name} not supported")
            else:
                target_mean = torch.mean(edit_res[cls.test_crit]).item()
                out[test_name] = target_mean

        with open("test_edit.yaml", "w") as f:
            yaml.dump(out, f)

        print(json.dumps(out, indent=4))

    def save_img_res(self, path: str, image: torch.Tensor) -> None:
        """Saves image result for debugging

        Args:
            path (str): Path to save image
            image (torch.Tensor): image tensor
        """
        img_inv = self.postproc(image)
        cv2.imwrite(str(path), cv2.cvtColor(img_inv, cv2.COLOR_RGB2BGR))

    def assert_mean_almost_equal(self, tensor, target_mean, places=None):
        output_mean = torch.mean(tensor).item()
        self.assertAlmostEqual(output_mean, target_mean, places=places)

    @classmethod
    def generate_test_functions(cls) -> None:
        """Automatically generate all test functions for editing.
        Checks if edit methods yield correct output image mean.
        """

        cls.setUpClass()

        for test_name, edit_cfg in cls.edit_cfgs.items():
            setattr(cls, f"test_{test_name}_equal", TestEdit.template_test_edit(test_name, edit_cfg, cls.test_img, cls.test_source_prompt, cls.test_target_prompt, cls.test_edit_method_configs.get(edit_cfg["edit"]["type"], None)))

    @staticmethod
    def template_test_edit(test_name: str, edit_config: Dict[str, Any], image: str, 
                           source_prompt: str, target_prompt: str, edit_add_config: Dict[str, Any]) -> Callable:
        """Generate test function which checks output of editing matches target mean.

        Args:
            test_name (str): Name of the test
            edit_config (Dict[str, Any]): Editing config to test
            image (str): Image path to test
            source_prompt (str): Source prompt
            target_prompt (str): Target prompt
            edit_add_config (Dict[str, Any]): Additional editing config (ptp)

        Returns:
            Callable: Test function
        """

        def test_func(self):
            print("\n", self.id())

            edit_res = self.edit_helper(edit_config, image, source_prompt, target_prompt, edit_add_config)

            if edit_res is None:
                # not supported
                return

            edit_image = edit_res[self.test_crit]

            if self.save_image_path is not None:
                self.save_image_path.mkdir(parents=True, exist_ok=True)
                self.save_img_res(self.save_image_path / (test_name + ".png"), edit_image)

            mean_target = self.test_data.get(test_name, None)

            self.assertIsNotNone(mean_target, f"No test data found. Output mean is {torch.mean(edit_image).item()}")
            self.assert_mean_almost_equal(edit_image, mean_target, places=None if edit_config["inversion"]["type"] not in ("nti",) else 3)
            # self.assert_mean_almost_equal(edit_res["latent"], target_mean)

        return test_func


if __name__ == '__main__':
    with torch.no_grad():
        if len(sys.argv) >= 2 and sys.argv[1] == "gen":
            # generate target data
            TestEdit.generate_test_data()
        else:
            # generate test functions
            TestEdit.generate_test_functions()
            unittest.main()
