from utils.debug_utils import enable_deterministic
enable_deterministic()

import torch

from modules import load_inverter, load_editor
from modules import StablePreprocess, StablePostProc
from diffusers import StableDiffusionPipeline
from typing import Dict, Any, List


# 
def dict_set_deep(dic: Dict[str, Any], key: str, val: Any) -> None:
    """Sets a key with "." separators in a nested dict.

    Args:
        dic (Dict[str, Any]): Dict to modify
        key (str): Key with "."
        val (Any): Value to set
    """

    def _set(dic: Dict[str, Any], keys: List[str]) -> None:
        key, keys = keys[0], keys[1:]

        if len(keys) == 0:
            dic[key] = val
        else:
            if key not in dic:
                dic[key] = {}

            _set(dic[key], keys)

    _set(dic, key.split("."))


def to_nested_dict(dic: Dict[str, Any]) -> Dict[str, Any]:
    """Helper functions to convert a flat dict with "." in keys to a nested dict

    Args:
        dic (Dict[str, Any]): Flat dict where keys contain "." separators.

    Returns:
        Dict[str, Any]: Nested dict.
    """

    out = {}

    for k, v in dic.items():
        dict_set_deep(out, k, v)

    return out


def dict_equal(dic1: Dict[str, Any], dic2: Dict[str, Any]) -> bool:
    """Test if two dicts are equal.

    Args:
        dic1 (Dict[str, Any]): First dict.
        dic2 (Dict[str, Any]): Second dict.

    Returns:
        bool: True if equal, otherwise False.
    """

    if dic1 is None or dic2 is None:
        return False

    for k, v in dic1.items():
        if k not in dic2 or v != dic2[k]:
            return False
        
    return True


class EditorManager:
    """Class processing editing requests from Gradio output
    """

    def __init__(self) -> None:
        # cached components to speed up inference
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

    def process_ptp_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Formats ptp config for ptp editor

        Args:
            cfg (Dict[str, Any): Config from gradio

        Returns:
            Dict[str, Any: Config with formatted ptp options.
        """

        ptp = cfg["editor"]["methods"]["ptp"]
        ptp_dft_cfg = ptp["dft_cfg"]

        ptp["dft_cfg"] = {
            "is_replace_controller": ptp_dft_cfg["is_replace_controller"],
            "cross_replace_steps": {'default_': ptp_dft_cfg["cross_replace_steps"]},
            "self_replace_steps": ptp_dft_cfg["self_replace_steps"],
            "blend_words": (
                (ptp_dft_cfg["source_blend_word"],), 
                (ptp_dft_cfg["target_blend_word"],)
            ),
            "equilizer_params": {
                "words": (ptp_dft_cfg["eq_params_words"],),
                "values": (ptp_dft_cfg["eq_params_values"],),
            },
        }

        return cfg

    @torch.no_grad()
    def run(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Runs editing for the passed config from Gradio and returns edited image results

        Args:
            cfg (Dict[str, Any]): Config from Gradio

        Returns:
            Dict[str, Any]: Image editing result.
        """

        # convert flat to nested dict
        cfg = to_nested_dict(cfg)

        # format ptp config
        cfg = self.process_ptp_config(cfg)

        # get inverter and editor arguments for the selected inverter and editor
        cfg["inverter"].update(cfg["inverter"]["methods"][cfg["inverter"]["type"]])
        del cfg["inverter"]["methods"]

        cfg["editor"].update(cfg["editor"]["methods"].get(cfg["editor"]["type"], {}))
        del cfg["editor"]["methods"]

        # get basic editing arguments
        source_image = cfg["editor"].pop("source_image")
        source_prompt = cfg["editor"].pop("source_prompt")
        target_prompt = cfg["editor"].pop("target_prompt")

        # reload components if config changed
        if not dict_equal(cfg["model"], self.cfg.get("model", None)):
            self.model = StableDiffusionPipeline.from_pretrained(cfg["model"]["type"]).to(self.device)
            self.preproc = StablePreprocess(self.device, size=512, center_crop=True, return_np=False, pil_resize=True)
            self.postproc = StablePostProc()
            self.cfg["inverter"] = None

        if not dict_equal(cfg["inverter"], self.cfg.get("inverter", None)):
            self.inverter = load_inverter(model=self.model, **cfg["inverter"])
            self.cfg["editor"] = None

        if not dict_equal(cfg["editor"], self.cfg.get("editor", None)):
            self.editor = load_editor(inverter=self.inverter, **cfg["editor"])

        # editing
        enable_deterministic()
        image = self.preproc(source_image)
        edit_res = self.editor.edit(image, source_prompt, target_prompt)
        img_edit = self.postproc(edit_res["image"])

        self.cfg = cfg

        return {
            "edit_image": img_edit
        }