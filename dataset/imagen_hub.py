import numpy as np

from datasets import load_dataset
from typing import Dict, Any

from .base import DatasetBase

class ImagenHubData(DatasetBase):
    """Class for ImagenHub dataset. Refer to https://tiger-ai-lab.github.io/ImagenHub/
    """

    def __init__(self, skip_img_load: bool=False, limit: int=None, split: str="dev", img_size: int=512) -> None:
        """Instantiate a new ImagenHub dataset.

        Args:
            skip_img_load (bool, optional): If set to true, images are not loaded and only their paths are returned. Defaults to False. Defaults to False.
            limit (int, optional): Optioally cut off dataset after limit. Defaults to None. 
            split (str, optional): Split to load. Defaults to "dev". 
            img_size (int, optional): Image and mask will be resized to this size. Defaults to 512.
        """

        if skip_img_load:
            print(f"skip_img_load is not supported")

        # TODO: remove cache path
        # Load dataset from huggingface
        self.data = load_dataset("ImagenHub/Text_Guided_Image_Editing")
        self.split = split
        self.limit = limit
        self.skip_img_load = False
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.data[self.split]) if self.limit is None else self.limit

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load a item from the dataset.

        Args:
            idx (int): Index of the item to load.

        Returns:
            Dict[str, Any]: Loaded item
        """
        
        sample = self.data[self.split][idx]

        image = np.array(sample["source_img"].resize((self.img_size, self.img_size)))

        # mask is a smooth mask (not binary) stored in the alpha channel
        # mask is stored as background mask. invert to forground mask
        mask = 1 - (np.array(sample["mask_img"].split()[-1].resize((self.img_size, self.img_size)), np.float32) / 255)

        source_prompt = sample["source_global_caption"]
        target_prompt = sample["target_global_caption"]
        name = f'{sample["img_id"]}_{source_prompt}_{target_prompt}'

        out = {
            "name": name,
            "image": image,
            "image_file": None,
            "source_prompt": source_prompt,
            "target_prompt": target_prompt,
            "mask": mask,
            "edit": {
                "target_prompt": target_prompt,
                "ptp": None,  # ptp not supported
            }
        }

        return out
