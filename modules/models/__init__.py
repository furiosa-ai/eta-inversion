
import torch
import numpy as np
import cv2
from pathlib import Path
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image
import diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion
from typing import Tuple, Union, Optional, Dict, Any


class StablePreprocess:
    """Class for Stable Diffusion preprocessing. Converts a image file or np.uint8 array for SD's VAE.
    """

    def __init__(self, device: str, size: int=512, return_np: bool=False, center_crop: bool=False, pil_resize: bool=False) -> None:
        """Initializes a new preprocessing functor for Stable Diffusion.

        Args:
            device (str): Which device to use.
            size (int, optional): Resize image size. Defaults to 512.
            return_np (bool, optional): If True, __call_ will return both a torch.Tensor and a uint8 np.ndarray. Otherwise only a torch.Tensor will be returned. Defaults to False.
            center_crop (bool, optional): If true, center crop will be used to preserve the aspect ratio of the image. Defaults to False.
            pil_resize (bool, optional): If true PIL Image library will be used for resizing, otherwise cv2 will be used. Defaults to False.
        """
        self.device = device
        self.size = size
        self.return_np = return_np
        self.center_crop = center_crop
        self.pil_resize = pil_resize

    def __call__(self, image: Union[str, Path, np.ndarray]) -> Union[torch.Tensor, Tuple[torch.Tensor, np.ndarray]]:
        """Preprocesses an image file or array for Stable Diffusion. Converts a image file or np.uint8 array for SD's VAE.

        Args:
            image (str): Path to an image file or image array

        Returns:
            torch.Tensor: _description_
        """

        if isinstance(image, (str, Path)):
            # load image
            image = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)

        if self.center_crop:
            h, w = image.shape[:2]

            # find larger size and crop to square
            if w > h:
                x1 = (w - h) // 2
                x2 = w - h - x1

                if x2 > 0:
                    image = image[:, x1:-x2]
            else:
                y1 = (h - w) // 2
                y2 = h - w - y1

                if y2 > 0:
                    image = image[y1:-y2]
        
        # use PIL or cv2 for resize. Prompt-to-prompt, ... mostly use PIL
        if self.pil_resize:
            image = np.array(Image.fromarray(image).resize((self.size, self.size)))
        else:
            image = cv2.resize(image, (self.size, self.size))
        
        # normalize to [-1, 1] and permute
        image_pt = torch.from_numpy(image).float() / 127.5 - 1
        image_pt = image_pt.permute(2, 0, 1).unsqueeze(0).to(self.device)

        if self.return_np:
            return image_pt, image
        else:
            return image_pt


class StablePostProc:
    """Class for Stable Diffusion postprocessing. Converts SD's VAE output to a np.uint8 array.
    """

    def __init__(self) -> None:
        """Initializes a new postprocessing functor for Stable Diffusion.
        """
        pass

    def __call__(self, image: torch.Tensor) -> np.ndarray:
        """Converts SD's VAE output to a np.uint8 array.

        Args:
            image (torch.Tensor): VAE output

        Returns:
            np.ndarray: Output image
        """

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).astype(np.uint8)
        return image[0]


def load_diffusion_model(model: str="CompVis/stable-diffusion-v1-4", device: str="cuda", 
                         preproc_args: Optional[Dict[str, Any]]=None,
                         ) -> Tuple[StableDiffusionPipeline, StablePreprocess, StablePostProc]:
    """Loads a diffusion model from HuggingFace with respective preprocessing and postprocessing function.

    Args:
        model (str, optional): Model name to load. Defaults to "CompVis/stable-diffusion-v1-4".
        device (str, optional): Device to use. Defaults to "cuda".
        preproc_args (Optional[Dict[str, Any]], optional): Arguments passed to the preprocessing function. Defaults to None.

    Returns:
        Tuple[StableDiffusionPipeline, StablePreprocess, StablePostProc]: Model pipeline, preprocessing function, postprocessing function
    """
    
    print(f"Loading model {model} ...")

    if model in ("CompVis/stable-diffusion-v1-4",):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        model = StableDiffusionPipeline.from_pretrained(model, scheduler=scheduler).to(device)
        return model, (StablePreprocess(device, size=512, **(preproc_args if preproc_args is not None else {})), StablePostProc())
    else:
        raise Exception(model)
