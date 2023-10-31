# Refer to https://github.com/timothybrooks/instruct-pix2pix/blob/main/metrics/clip_similarity.py

from __future__ import annotations

import clip
import torch
import torch.nn.functional as F
from einops import rearrange

from .base import SimpleMetric
from typing import Tuple


class ClipSimilarityPix2Pix(SimpleMetric):
    """Ranges from 1 (best) to 0 (worst). Uses CLIP to obtain text embeddings for source and target prompt and image embeddings for source and output image.
    Computes similarity of direction from source to output image and direction from source to target prompt
    as (clip(output_image) - clip(source_image)) @ (clip(target_prompt) - clip(source_prompt)).
    """

    def __init__(self, input_range: Tuple[int, int]=(-1, 1), device: None=None, name: str = "ViT-L/14") -> None:
        """Initializes a new metric object.

        Args:
            input_range (Tuple[int, int], optional): Input range for image tensors needed for normalization. Defaults to (-1, 1).
            device (Optional[str], optional): Device to compute the metric on. Defaults to None.
            name (str, optional): Name of the CLIP model to use. Defaults to "ViT-L/14".
        """
        super().__init__(input_range=input_range, device=device)

        super().__init__()
        assert name in ("RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px")  # fmt: skip

        # input sizes of various clip models
        self.size = {"RN50x4": 288, "RN50x16": 384, "RN50x64": 448, "ViT-L/14@336px": 336}.get(name, 224)

        self.model, _ = clip.load(name, device=self.device, download_root="./")
        self.model.eval().requires_grad_(False)

        self.mean = torch.tensor((0.48145466, 0.4578275, 0.40821073), device=self.device)
        self.std = torch.tensor((0.26862954, 0.26130258, 0.27577711), device=self.device)

    def encode_text(self, text: list[str]) -> torch.Tensor:
        """Encodes text

        Args:
            text (list[str]): List of texts to encode

        Returns:
            torch.Tensor: Normalized text embedding
        """

        text = clip.tokenize(text, truncate=True).to(self.device)
        text_features = self.model.encode_text(text)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:  # Input images in range [0, 1].
        """Encodes image

        Args:
            image (torch.Tensor): Image tesnor

        Returns:
            torch.Tensor: Normalized image embedding
        """

        # resize to correct input size
        image = F.interpolate(image.float(), size=self.size, mode="bicubic", align_corners=False)
        image = image - rearrange(self.mean, "c -> 1 c 1 1")
        image = image / rearrange(self.std, "c -> 1 c 1 1")
        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def forward(self, source_image: torch.Tensor, target_image: torch.Tensor, source_prompt: str, target_prompt: str) -> torch.Tensor:
        """Computes metric value as (clip(output_image) - clip(source_image)) @ (clip(target_prompt) - clip(source_prompt)).

        Args:
            source_image (torch.Tensor): Source image
            target_image (torch.Tensor): Output image
            source_prompt (str): Source prompt
            target_prompt (str): Target prompt

        Returns:
            torch.Tensor: Metric value
        """

        source_image = self._normalize(source_image)
        target_image = self._normalize(target_image)

        image_features_0 = self.encode_image(source_image)
        image_features_1 = self.encode_image(target_image)
        text_features_0 = self.encode_text(source_prompt)
        text_features_1 = self.encode_text(target_prompt)
        # sim_0 = F.cosine_similarity(image_features_0, text_features_0)
        # sim_1 = F.cosine_similarity(image_features_1, text_features_1)
        sim_direction = F.cosine_similarity(image_features_1 - image_features_0, text_features_1 - text_features_0)
        # sim_image = F.cosine_similarity(image_features_0, image_features_1)
        # return sim_0, sim_1, sim_direction, sim_image
        return sim_direction

    def __repr__(self) -> str:
        return "clip_pix2pix"