import json
import os
import numpy as np
import copy
from PIL import Image
import torch
from typing import Dict, List, Any

from dataset.base import DatasetBase


class PieBenchData(DatasetBase):
    # PIE category names and image indices
    categories = {
        '0_random': range(0, 140),
        '1_change_object': range(140, 220),
        '2_add_object': range(220, 300),
        '3_delete_object': range(300, 380),
        '4_change_attribute_content': range(380, 420),
        '5_change_attribute_pose': range(420, 460),
        '6_change_attribute_color': range(460, 500),
        '7_change_attribute_material': range(500, 540),
        '8_change_background': range(540, 620),
        '9_change_style': range(620, 700)
    }

    """Class for PIE (DirectInversion) dataset. Refer to https://github.com/cure-lab/DirectInversion
    """

    def __init__(self, data_path: str="data/eval/PIE-Bench_v1", skip_img_load: bool=False, limit: int=None, categories=None) -> None:
        """Instantiate a new PIE dataset.

        Args:
            data_path (str, optional): Path to dataset directory. Defaults to "data/eval/PIE-Bench_v1".
            skip_img_load (bool, optional): If set to true, images are not loaded and only their paths are returned. Defaults to False.
            limit (int, optional): Optioally cut off dataset after limit. Defaults to None.
        """
        
        # load data from mapping_file.json
        with open(f"{data_path}/mapping_file.json", "r") as f:
            editing_instruction = json.load(f)
        
        labels = []

        for key, item in editing_instruction.items():
            # load prompts, remove special characters
            original_prompt = item["original_prompt"].replace("[", "").replace("]", "")
            editing_prompt = item["editing_prompt"].replace("[", "").replace("]", "")
            image_path = os.path.join(f"{data_path}/annotation_images", item["image_path"])
            editing_instruction = item["editing_instruction"]

            # for ptp
            blended_word = item["blended_word"].split(" ") if item["blended_word"] != "" else []
            
            # fg mask (needed to certain loss functions such as bglpips)
            mask = item["mask"]

            # default ptp config used in PIE benchmark
            ptp_cfg = dict(
                is_replace_controller=False,
                prompts = [original_prompt, editing_prompt],
                cross_replace_steps={'default_': .4,},
                self_replace_steps=0.6,
                blend_words=(((blended_word[0], ),
                            (blended_word[1], ))) if len(blended_word) else None,
                equilizer_params={
                    "words": (blended_word[1], ),
                    "values": (2, )
                } if len(blended_word) else None,
            )

            labels.append(dict(
                name=image_path,
                source_prompt=original_prompt,
                target_prompt=editing_prompt,
                image_file=image_path,
                edit=dict(
                    target_prompt=editing_prompt,
                    ptp=ptp_cfg,
                ),
                mask=mask,
            ))
        
        if categories is not None:
            ind = sum([list(PieBenchData.categories[cat]) for cat in categories], [])
            labels = [labels[i] for i in ind]

        self.edit_prompts = labels
        self.skip_img_load = skip_img_load
        self.limit = limit

    def mask_decode(self, encoded_mask: List[int], image_shape: List[int]=[512,512]) -> torch.Tensor:
        length=image_shape[0]*image_shape[1]
        mask_array=np.zeros((length,), dtype=np.float32)
        
        for i in range(0,len(encoded_mask),2):
            splice_len=min(encoded_mask[i+1],length-encoded_mask[i])
            for j in range(splice_len):
                mask_array[encoded_mask[i]+j]=1
                
        mask_array=mask_array.reshape(image_shape[0], image_shape[1])
        # to avoid annotation errors in boundary
        mask_array[0,:]=1
        mask_array[-1,:]=1
        mask_array[:,0]=1
        mask_array[:,-1]=1
                
        return torch.from_numpy(mask_array)

    def __len__(self) -> int:
        return len(self.edit_prompts) if self.limit is None else self.limit

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load a item from the dataset.

        Args:
            idx (int): Index of the item to load.

        Returns:
            Dict[str, Any]: Loaded item
        """

        edit_prompt = self.edit_prompts[idx]

        image = np.array(Image.open(edit_prompt["image_file"]))[:, :, :3] if not self.skip_img_load else None
        mask = self.mask_decode(edit_prompt["mask"])

        if edit_prompt["edit"]["ptp"]["blend_words"] is not None:
            edit_word_src = edit_prompt["edit"]["ptp"]["blend_words"][0][0]
            edit_word_target = edit_prompt["edit"]["ptp"]["blend_words"][1][0]
        else:
            edit_word_src, edit_word_target = None, None

        source_prompt = edit_prompt["edit"]["ptp"]["prompts"][0]
        target_prompt = edit_prompt["edit"]["ptp"]["prompts"][1]

        edit_word_idx = [None, None]

        try:
            edit_word_idx[0] = source_prompt.split(" ").index(edit_word_src)
        except ValueError:
            edit_word_src = None

        try:
            edit_word_idx[1] = target_prompt.split(" ").index(edit_word_target)
        except ValueError:
            edit_word_target = None

        # edit_word_idx = (source_prompt.split(" ").index(edit_word_src), target_prompt.split(" ").index(edit_word_target))
        
        out = {
            **copy.deepcopy(edit_prompt),
            "image": image,
            "mask": mask,
            "edit_word_idx": edit_word_idx,
        }

        return out

    def __repr__(self) -> str:
        return "pie"
