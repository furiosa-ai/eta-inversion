from .base import SimpleMetric
import numpy as np
import torch
from PIL import Image

from .clip_similarity import imagenet_templates, get_embedding_for_prompt
from typing import Tuple, Optional


class BLIPCLIPSimilarity(SimpleMetric):
    """Text-text CLIP similarity with BLIP image caption and prompt. Ranges from 1 (best) to 0 (worst). Refer to
    https://github.com/yuval-alaluf/Attend-and-Excite/blob/main/metrics/blip_captioning_and_clip_similarity.py
    """

    def __init__(self, input_range: Tuple[int, int]=(-1, 1), device: Optional[str]=None, use_imagenet_templates: bool=False) -> None:
        """Initializes a new BLIPCLIP similarity metric.

        Args:
            input_range (Tuple[int, int], optional): Input range for image tensors needed for normalization. Defaults to (-1, 1).
            device (Optional[str], optional): Device to compute the metric on. Defaults to None.
            use_imagenet_templates (bool, optional): If True CLIP text embeddings for prompt will be averaged over 80 imagenet templates. Defaults to False.
        """
        super().__init__(input_range, device)

        import clip
        from lavis.models import load_model_and_preprocess

        # load clip
        self.clip, self.clip_preproc = clip.load("ViT-B/16", self.device)

        # load blip
        self.blip, self.blip_preproc, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco",
                                                              is_eval=True, device=self.device)

        self.templates = imagenet_templates if use_imagenet_templates else ["{}"]
        self.losses = []

    def forward(self, pred: torch.Tensor, target: str) -> torch.Tensor:
        """Compute metric value for a single example.

        Args:
            pred (torch.Tensor): Output image
            target (torch.Tensor): Target prompt

        Returns:
            torch.Tensor: Metric value (1 best, 0 worst)
        """

        import clip
        
        image, prompt = pred, target

        # image = self._normalize(image)
        if not isinstance(image, np.ndarray):
            image = np.clip((self._normalize(image)[0].permute(1, 2, 0)).detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)

        # encode prompt
        prompt_features = get_embedding_for_prompt(self.clip, prompt, templates=self.templates)

        # extract blip captions and embeddings
        image = Image.fromarray(image)
        blip_input_image = self.blip_preproc["eval"](image).unsqueeze(0).to(self.device)
        blip_caption = self.blip.generate({"image": blip_input_image})[0]
        text = clip.tokenize([blip_caption]).to(self.device)
        caption_embedding = self.clip.encode_text(text)
        caption_embedding = caption_embedding / caption_embedding.norm(dim=-1, keepdim=True)

        # compute text-text similarity between predicted BLIP text embeddings and prompt embeddings
        sim = caption_embedding.float() @ prompt_features

        return sim

    def __repr__(self) -> str:
        return "blipclip"


class BLIPCLIPAccuracy(SimpleMetric):
    """Similar to CLIPAccuracy. Ranges from 1 (best) to 0 (worst).
    Computes the BLIPCLIP imilarity of the output image with the source prompt and the target prompt
    and classifies an image as correctly edited if the similarity with the target prompt is higher than with the source prompt.
    The final metric is the ratio of corectly edited images by the total number of images.
    """

    def __init__(self, input_range: Tuple[int, int]=(-1, 1), device: Optional[str]=None) -> None:
        super().__init__(input_range, device)

        self.blipclip_sim = BLIPCLIPSimilarity(input_range, device)

    def forward(self, image: torch.Tensor, source_prompt: str, target_prompt: str) -> torch.Tensor:
        """Compute metric value for a single example.

        Args:
            image (torch.Tensor): Source image/dited output image
            source_prompt (str): Prompt used for inversion.
            target_prompt (str): Prompt used for editing/denoising.

        Returns:
            torch.Tensor: Metric value (1 best, 0 worst)
        """
        
        sim_src = self.blipclip_sim(image, source_prompt)
        sim_target = self.blipclip_sim(image, target_prompt)
        val = sim_target > sim_src

        return val.float()

    def __repr__(self) -> str:
        return "blipclip_acc"

