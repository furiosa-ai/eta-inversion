from utils.debug_utils import enable_deterministic
enable_deterministic()

import torch
from pathlib import Path
import cv2
import argparse
from tqdm import trange

from modules import get_edit_methods, get_inversion_methods, load_inverter, load_editor
from modules.inversion.diffusion_inversion import DiffusionInversion
# from modules.exceptions import DiffusionInversionException
from utils.eval_utils import EditResultData
from modules import StablePreprocess, StablePostProc
from diffusers import StableDiffusionPipeline
from typing import List, Tuple

from utils.utils import add_argparse_arg


def dict_get(dic, key, dft=None):
    def _get(dic, keys):
        if len(keys) == 0:
            return dic
        else:
            key, keys = keys[0], keys[1:]

            if key not in dic:
                return dft
            else:
                return _get(dic[key], keys)
            
    return _get(dic, key.split("."))


def dict_set_deep(dic, key, val):
    def _set(dic, keys):
        key, keys = keys[0], keys[1:]

        if len(keys) == 0:
            dic[key] = val
        else:
            if key not in dic:
                dic[key] = {}

            _set(dic[key], keys)

    _set(dic, key.split("."))


def to_nested_dict(dic):
    out = {}

    for k, v in dic.items():
        dict_set_deep(out, k, v)

    return out

def _dict_equal(dic1, dic2):
    if dic1 is None or dic2 is None:
        return False

    for k, v in dic1.items():
        if k not in dic2 or v != dic2[k]:
            return False
        
    return True


class EditorManager:
    def __init__(self) -> None:
        self.cached = {
            "model": None,
            "inverter": None,
            "editor": None,
        }

        self.device = "cuda"
        self.model = None
        self.preproc = None
        self.postproc = None
        self.inverter = None
        self.editor = None
        self.cfg = {}

    @torch.no_grad()
    def run(self, cfg):
        cfg = to_nested_dict(cfg)

        cfg["inverter"].update(cfg["inverter"]["methods"][cfg["inverter"]["type"]])
        del cfg["inverter"]["methods"]

        # cfg["editor"].update(cfg["editor"]["methods"][cfg["editor"]["type"]])
        # del cfg["editor"]["methods"]

        if not _dict_equal(cfg["model"], self.cfg.get("model", None)):
            self.model = StableDiffusionPipeline.from_pretrained(cfg["model"]["type"]).to(self.device)
            self.preproc = StablePreprocess(self.device, size=512, center_crop=True, return_np=False, pil_resize=True)
            self.postproc = StablePostProc()
            self.cfg["inverter"] = None

        if not _dict_equal(cfg["inverter"], self.cfg.get("inverter", None)):
            self.inverter = load_inverter(model=self.model, **cfg["inverter"])
            self.cfg["editor"] = None

        if not _dict_equal(cfg["editor"], self.cfg.get("editor", None)):
            self.editor = load_editor(inverter=self.inverter, **cfg["editor"])

        source_image = cfg["edit_cfg"].pop("source_image")
        source_prompt = cfg["edit_cfg"].pop("source_prompt")
        target_prompt = cfg["edit_cfg"].pop("target_prompt")

        image = self.preproc(source_image)
        edit_res = self.editor.edit(image, source_prompt, target_prompt, cfg["edit_cfg"] if len(cfg["edit_cfg"]) > 0 else None)  #  cfg=self.get_key("edit_cfg")
        img_edit = self.postproc(edit_res["image"])

        self.cfg = cfg

        return {
            "edit_image": img_edit
        }

        # if model != self.cached["model"]:
        #     del self.model
        #     self.model = StableDiffusionPipeline.from_pretrained(model).to(self.device)
        #     self.preproc = StablePreprocess(self.device, size=512, center_crop=True, return_np=False, pil_resize=True)
        #     self.postproc = StablePostProc()
        #     self.cached["model"] = model

        # if inverter != self.cached["inverter"]:
        #     self.inverter = load_inverter(model=self.model, 
        #                                   type=inverter, 
        #                                 #   scheduler=self.get_key("inverter.scheduler"), 
        #                                 #   num_inference_steps=self.get_key("inverter.num_inference_steps"), 
        #                                 #   guidance_scale_fwd=self.get_key("inverter.guidance_scale_fwd"),
        #                                 #   guidance_scale_bwd=self.get_key("inverter.guidance_scale_bwd"),
        #                                 #   **self.get_key("inverter.cfg")
        #                                   )
            
        #     self.cached["inverter"] = inverter

        # if editor != self.cached["editor"]:
        #     self.editor = load_editor(inverter=self.inverter, type=editor)
        #     self.cached["editor"] = None

        # image = self.preproc(source_image)
        # edit_res = self.editor.edit(image, source_prompt, target_prompt)  #  cfg=self.get_key("edit_cfg")
        # img_edit = self.postproc(edit_res["image"])




def main():
    class InventoryItem:
        """Class for keeping track of an item in inventory."""
        name: str
        unit_price: float
        quantity_on_hand: int = 0

    item = InventoryItem()

    print(type(item.name))


if __name__ == "__main__":
    main()
