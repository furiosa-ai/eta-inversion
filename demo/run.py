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
