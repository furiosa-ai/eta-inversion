import json
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Any
import random

from dataset.base import DatasetBase

# from .base import DatasetBase

class CocoData(DatasetBase):
    """Class for COCO dataset. Used for reconstruction performance evaluation.
    """

    def __init__(self, data_path: str="data/eval/coco", skip_img_load: bool=False, limit: int=100, split="train2017") -> None:
        """Instantiate a new COCO dataset.

        Args:
            data_path (str, optional): Path to dataset directory. Defaults to "data/eval/PIE-Bench_v1".
            skip_img_load (bool, optional): If set to true, images are not loaded and only their paths are returned. Defaults to False.
            limit (int, optional): Optioally cut off dataset after limit. Defaults to None.
            split (str, optional): COCO split to use. Defaults to "train2017".
        """

        img_dir = Path(data_path) / split
        label_file = Path(data_path) / "annotations" / f"captions_{split}.json"
        
        with open(label_file, "r") as f:
            labels = json.load(f)

        # deterministic shuffle
        random.Random(0).shuffle(labels["annotations"])

        self.img_files = []
        self.captions = []

        # load image paths and captions
        for anno in labels["annotations"][:limit]:
            img_id = anno["image_id"]
            caption = anno["caption"]
            img_file = img_dir / f"{img_id:012d}.jpg"

            assert img_file.exists()

            self.img_files.append(img_file)
            self.captions.append(caption)
        
        self.skip_img_load = skip_img_load
        self.limit = limit

    def __len__(self) -> int:
        return len(self.img_files) if self.limit is None else self.limit

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load a item from the dataset.

        Args:
            idx (int): Index of the item to load.

        Returns:
            Dict[str, Any]: Loaded item
        """

        image = np.array(Image.open(self.img_files[idx]))[:, :, :3] if not self.skip_img_load else None
        caption = self.captions[idx]

        out = {
            "name": caption,
            "image": image,
            "image_file": str(self.img_files[idx]),
            "source_prompt": caption,
            "target_prompt": "",
            "edit": {
                "target_prompt": "",
            },
            "mask": None,
        }

        return out

    def __repr__(self) -> str:
        return "coco"
