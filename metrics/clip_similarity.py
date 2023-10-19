

from .base import SimpleMetric
import numpy as np
import torch
from PIL import Image
from clip.model import CLIP
from typing import List, Tuple, Optional


# Imagenet caption templates. Used to obtain better CLIP features for a prompt by averaging over all templates
# Refer to https://github.com/yuval-alaluf/Attend-and-Excite/blob/main/metrics/compute_clip_similarity.py
imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]


def get_embedding_for_prompt(model: CLIP, prompt: str, templates: List[str]) -> torch.Tensor:
    """Compute feature embeddings for the prompt using CLIP. Feature embeddings will be averaged over all templates.

    Args:
        model (CLIP): CLIP model
        prompt (str): Prompt for feature extraction
        templates (List[str]): Prompt will be inserted into provided templates and the final embedding will be averaged over all templates.

    Returns:
        torch.Tensor: Embedding
    """

    import clip

    # create captions to average using templates
    texts = [template.format(prompt) for template in templates]  # format with class
    texts = [t.replace('a a', 'a') for t in texts]  # remove double a's
    texts = [t.replace('the a', 'a') for t in texts]  # remove double a's
    texts = clip.tokenize(texts).cuda()  # tokenize

    # encode and normalize for each caption
    class_embeddings = model.encode_text(texts)  # embed with text encoder
    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

    # compute mean embedding and normalize again
    class_embedding = class_embeddings.mean(dim=0)
    class_embedding /= class_embedding.norm()
    return class_embedding.float()


class CLIPSimilarity(SimpleMetric):
    """CLIP similarity for image-text similarity. Ranges from 1 (best) to 0 (worst)."""

    def __init__(self, input_range: Tuple[int, int]=(-1, 1), device: Optional[str]=None, use_imagenet_templates: bool=False) -> None:
        """Initializes a new CLIP similarity metric.

        Args:
            input_range (Tuple[int, int], optional): Input range for image tensors needed for normalization. Defaults to (-1, 1).
            device (Optional[str], optional): Device to compute the metric on. Defaults to None.
            use_imagenet_templates (bool, optional): If True CLIP text embeddings will be averaged over 80 imagenet templates. Defaults to False.
        """

        super().__init__(input_range, device)
        import clip

        # load clip
        self.model, self.preprocess = clip.load("ViT-B/16", self.device)
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

        image, prompt = pred, target

        # image = self._normalize(image)
        if not isinstance(image, np.ndarray):
            image = np.clip((self._normalize(image)[0].permute(1, 2, 0)).detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)

        image = Image.fromarray(image)
        image_preproc = self.preprocess(image).unsqueeze(0).to(self.device)

        # encode prompt
        text_feat = get_embedding_for_prompt(self.model, prompt, templates=self.templates)

        # encode image and normalize embedding
        img_feat = self.model.encode_image(image_preproc)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)

        # dot product
        sim = (img_feat.float() @ text_feat.T)

        return sim

    def __repr__(self) -> str:
        return "clip"


class CLIPAccuracy(SimpleMetric):
    """CLIP accuracy from pix2pix-zero (https://arxiv.org/abs/2302.03027). Ranges from 1 (best) to 0 (worst).
    Computes the CLIP image-text similarity of the output image with the source prompt and the target prompt
    and classifies an image as correctly edited if the similarity with the target prompt is higher than with the source prompt.
    The final metric is the ratio of corectly edited images by the total number of images.
    """

    def __init__(self, input_range: Tuple[int, int]=(-1, 1), device: Optional[str]=None) -> None:
        super().__init__(input_range, device)

        self.clip_sim = CLIPSimilarity(input_range, device)

    def forward(self, image: torch.Tensor, source_prompt: str, target_prompt: str) -> torch.Tensor:
        """Compute metric value for a single example.

        Args:
            image (torch.Tensor): Source image/dited output image
            source_prompt (str): Prompt used for inversion.
            target_prompt (str): Prompt used for editing/denoising.

        Returns:
            torch.Tensor: Metric value (1 best, 0 worst)
        """
        
        sim_src = self.clip_sim(image, source_prompt)
        sim_target = self.clip_sim(image, target_prompt)
        val = sim_target > sim_src

        return val.float()

    def __repr__(self) -> str:
        return "clip_acc"
