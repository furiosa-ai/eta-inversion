import sys
from pathlib import Path
from typing import Any, Callable, Dict
from modules.models import load_diffusion_model

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.debug_utils import enable_deterministic
enable_deterministic()

from itertools import product
import unittest

import torch
import cv2
import yaml
import json
from modules import load_editor, load_inverter


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
        dict(type="dirinv", scheduler="ddim", num_inference_steps=steps),
        dict(type="etainv", scheduler="ddim", num_inference_steps=steps),
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
        "diffinv_ddim_50__simple": -0.06134095788002014,                                                                                                                                                
        "diffinv_ddim_50__ptp": -0.046193402260541916,                                                                                                                                                  
        "diffinv_ddim_50__masactrl": -0.052706025540828705,                                                                                                                                             
        "diffinv_ddim_50__pnp": -0.006986429449170828,                                                                                                                                                  
        "diffinv_ddim_50__pix2pix_zero": -0.10092967748641968,                                                                                                                                          
        "nti_ddim_50__simple": -0.029194405302405357,                                                                                                                                                   
        "nti_ddim_50__ptp": -0.04410939663648605,                                                                                                                                                       
        "nti_ddim_50__masactrl": -0.052361153066158295,                                                                                                                                                 
        "nti_ddim_50__pnp": -0.06067529320716858,                                                                                                                                                       
        "nti_ddim_50__pix2pix_zero": -0.12025877088308334,                                                                                                                                              
        "npi_ddim_50__simple": -0.05122676491737366,                                                                                                                                                    
        "npi_ddim_50__ptp": -0.0405917689204216,                                                                                                                                                        
        "npi_ddim_50__masactrl": -0.052706025540828705,                                                                                                                                                 
        "npi_ddim_50__pnp": -0.06151632219552994,                                                                                                                                                       
        "npi_ddim_50__pix2pix_zero": -0.10092967748641968,                                                                                                                                              
        "proxnpi_ddim_50__simple": -0.046024784445762634,                                                                                                                                               
        "proxnpi_ddim_50__ptp": -0.041355475783348083,                                                                                                                                                  
        "proxnpi_ddim_50__masactrl": -0.0342615507543087,                                                                                                                                               
        "proxnpi_ddim_50__pnp": -0.04113534092903137,                                                                                                                                                   
        "proxnpi_ddim_50__pix2pix_zero": -0.024249499663710594,                                                                                                                                         
        "edict_ddim_50__simple": -0.04798874258995056,                                                                                                                                                  
        "edict_ddim_50__ptp": -0.03372775390744209,                                                                                                                                                     
        "edict_ddim_50__masactrl": -0.028166545554995537,
        "edict_ddim_50__pnp": -0.01832800731062889,
        "edict_ddim_50__pix2pix_zero": -0.035901546478271484,
        "ddpminv_ddpm_50__simple": -0.01691855490207672,
        "ddpminv_ddpm_50__ptp": -0.025561805814504623,
        "ddpminv_ddpm_50__masactrl": -0.011981125921010971,
        "ddpminv_ddpm_50__pnp": -0.013231728225946426,
        "ddpminv_ddpm_50__pix2pix_zero": -0.00701802596449852,

        "dirinv_ddim_50__simple": -0.06134095788002014,
        "dirinv_ddim_50__ptp": -0.0668606162071228,
        "dirinv_ddim_50__masactrl": -0.05110548064112663,
        "dirinv_ddim_50__pnp": -0.0034499389585107565,
        "dirinv_ddim_50__pix2pix_zero": 0.0,
        "etainv_ddim_50__simple": -0.0058508021757006645,
        "etainv_ddim_50__ptp": -0.026020852848887444,
        "etainv_ddim_50__masactrl": -0.007259005215018988,
        "etainv_ddim_50__pnp": 0.0134469298645854,
        "etainv_ddim_50__pix2pix_zero": 0.0,
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

        cls.model, (cls.preproc, cls.postproc) = load_diffusion_model("CompVis/stable-diffusion-v1-4", cls.device, variant="fp32", preproc_args={"center_crop": True, "return_np": False})

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
            setattr(cls, f"test_{test_name}_equal", TestEdit.template_test_edit(
                test_name, edit_cfg, cls.test_img, cls.test_source_prompt, cls.test_target_prompt, 
                cls.test_edit_method_configs.get(edit_cfg["edit"]["type"], None)))

            if edit_cfg["inversion"]["type"] == "diffinv":
                setattr(cls, f"test_{test_name}_consistency", TestEdit.template_test_edit_consistency(
                    test_name, edit_cfg, cls.test_img, cls.test_source_prompt, cls.test_target_prompt, 
                    cls.test_edit_method_configs.get(edit_cfg["edit"]["type"], None)))

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
    
    @staticmethod
    def template_test_edit_consistency(test_name: str, edit_config: Dict[str, Any], image: str, 
                           source_prompt: str, target_prompt: str, edit_add_config: Dict[str, Any]) -> Callable:
        """Generate test function which checks if editing output is consistent if run multiple times.

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

            edit_res1 = self.edit_helper(edit_config, image, source_prompt, target_prompt, edit_add_config)
            edit_res2 = self.edit_helper(edit_config, image, source_prompt + " test", target_prompt + " test", edit_add_config)
            edit_res3 = self.edit_helper(edit_config, image, source_prompt, target_prompt, edit_add_config)

            edit_image1 = edit_res1[self.test_crit]
            edit_image3 = edit_res3[self.test_crit]

            loss = torch.nn.MSELoss()(edit_image1, edit_image3).item()

            self.assertEqual(loss, 0)

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
