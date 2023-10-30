from utils.debug_utils import enable_deterministic
enable_deterministic()

import torch

from modules import load_inverter, load_editor
from modules import StablePreprocess, StablePostProc
from diffusers import StableDiffusionPipeline


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

    def process_ptp_config(self, cfg):
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
    def run(self, cfg):
        cfg = to_nested_dict(cfg)

        cfg = self.process_ptp_config(cfg)

        cfg["inverter"].update(cfg["inverter"]["methods"][cfg["inverter"]["type"]])
        del cfg["inverter"]["methods"]

        cfg["editor"].update(cfg["editor"]["methods"].get(cfg["editor"]["type"], {}))
        del cfg["editor"]["methods"]

        source_image = cfg["editor"].pop("source_image")
        source_prompt = cfg["editor"].pop("source_prompt")
        target_prompt = cfg["editor"].pop("target_prompt")

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

        image = self.preproc(source_image)
        edit_res = self.editor.edit(image, source_prompt, target_prompt)  #  cfg=self.get_key("edit_cfg")
        img_edit = self.postproc(edit_res["image"])

        self.cfg = cfg

        return {
            "edit_image": img_edit
        }