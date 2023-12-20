
from pathlib import Path
import cv2
import yaml
import pickle

from .base import DatasetBase
from typing import Any, Dict, Union


class EditingDataset(DatasetBase):
    """Main class for editing datasets. Loads information from a given yaml file.
    Can be used to load datasets like plug-and-play.
    """

    def __init__(self, path: str="data/eval/plug_and_play", skip_img_load: bool=False) -> None:
        """Instantiate a new dataset.

        Args:
            path (str, optional):  Path to dataset to load. Can be either a directory 
            containing a prompts.yaml file or a path to a yaml file. Defaults to "data/eval/plug_and_play". 
            skip_img_load (bool, optional): If set to true, images are not loaded and only their paths are returned. Defaults to False. 
        """
        super().__init__()

        prompt_path = Path(path)

        # If is directory, set to containing prompts.yaml
        if prompt_path.suffix != ".yaml":
            prompt_path /= "prompts.yaml"

        self.img_dir = prompt_path.parent / "imgs"
        self.skip_img_load = skip_img_load

        latents_path = prompt_path.parent / "latents.pkl"

        if latents_path.exists():
            with open(latents_path, "rb") as f:
                self.latents = pickle.load(f)
        else:
            self.latents = None

        with open(prompt_path, "r") as f:
            self.edit_prompts = yaml.safe_load(f)

    def __repr__(self) -> str:
        return "editingdata"

    def __len__(self) -> int:
        return len(self.edit_prompts)

    def _to_ptp(self, edit_prompt: Dict[str, Union[int, Any]]) -> Dict[str, Any]:
        """Generate a default configuration for prompt-to-prompt

        Args:
            edit_prompt (Dict[str, Union[int, Any]]): Current dataset item

        Returns:
            Dict[str, Any]: Default config for ptp
        """
        assert len(edit_prompt["edit"]) == 1, "Only one edit per prompt is supported"

        for edit in edit_prompt["edit"]:
            edit_type, edit_cfg = edit
            source_prompt, target_prompt = edit_prompt["source_prompt"], edit_prompt["target_prompt"]

            if edit_type == "replace":
                # if a word should be replaced
                (source_word, target_word) = edit_cfg

                assert " " not in source_word and " " not in target_word, "Edit word cannot contain space"

                # default config for replacing in ptp (from official ptp repo)
                return dict(
                    prompts = [source_prompt, target_prompt],
                    is_replace_controller = False,
                    cross_replace_steps = {'default_': .4,},
                    self_replace_steps = .6,
                    blend_words = (((source_word,), (target_word,))), # for local edit. If it is not local yet - use only the source object: blend_word = ((('cat',), ("cat",))).
                    equilizer_params = {"words": (target_word,), "values": (2,)}, # amplify attention to the word "tiger" by *2 
                )
            elif edit_type == "add":
                # if content should be added

                target_word = edit_cfg["word"].split(" ")
                blend_words = edit_cfg.get("blend_words", None)
                focus_words = edit_cfg.get("focus_words", None)

                # default config for adding in ptp (from official ptp repo)
                out = dict(
                    prompts = [source_prompt, target_prompt],
                    is_replace_controller = False,
                    cross_replace_steps = {'default_': .4,},
                    self_replace_steps = .6,
                    # blend_words = (((source_word,), (target_word,))), # for local edit. If it is not local yet - use only the source object: blend_word = ((('cat',), ("cat",))).
                    # equilizer_params = {"words": (target_word,), "values": (2,)}, # amplify attention to the word "tiger" by *2 
                )

                if blend_words is not None:
                    out["blend_words"] = (blend_words, blend_words)

                if focus_words is not None:
                    out["equilizer_params"] = {"words": focus_words, "values": (2 if blend_words is not None else 5,) * len(focus_words)}

                return out
            else:
                assert False
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load a item from the dataset.

        Args:
            idx (int): Index of the item to load.

        Returns:
            Dict[str, Any]: Loaded item
        """
        
        edit_prompt = self.edit_prompts[idx]

        # Load image if not skip_img_load
        image_file = self.img_dir / (edit_prompt["source_prompt"] + ".png")
        image = cv2.cvtColor(cv2.imread(str(image_file)), cv2.COLOR_BGR2RGB) if not self.skip_img_load else None

        out = {
            "name": edit_prompt["source_prompt"] + "-" + edit_prompt["target_prompt"],
            "image": image,
            "image_file": str(image_file),
            "source_prompt": edit_prompt["source_prompt"],
            "target_prompt": edit_prompt["target_prompt"],
            "edit": {
                "target_prompt": edit_prompt["target_prompt"],
                "ptp": self._to_ptp(edit_prompt),
            },
            "mask": None,
        }

        # Load groundtruth latent codes if available
        if self.latents is not None:
            out["zT_gt"] = self.latents[edit_prompt["source_prompt"]]

        return out
