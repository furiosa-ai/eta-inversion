

from .base import SimpleMetric
import numpy as np
import torch
from PIL import Image
from clip.model import CLIP
from typing import Any, List, Tuple, Optional, Union
import torch.nn.functional as F


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


class BLIPCaptionGenerator:
    """Generates text captions for images using BLIP
    """

    def __init__(self, device: str) -> None:
        """Instantiates a new BLIP caption generator

        Args:
            device (str): Device to run BLIP on
        """

        from lavis.models import load_model_and_preprocess

        self.device = device
        self.blip, self.blip_preproc, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco",
                                                              is_eval=True, device=self.device)

    def __call__(self, image: Image.Image) -> str:
        """Creates a caption for the provided image

        Args:
            image (Image.Image): Image to generate a caption for.

        Returns:
            str: Caption
        """

        blip_input_image = self.blip_preproc["eval"](image).unsqueeze(0).to(self.device)
        blip_caption = self.blip.generate({"image": blip_input_image})[0]
        return blip_caption


class CLIPSimilarity(SimpleMetric):
    """CLIP similarity for image-text similarity. Higher is better. Except for directional metric, ranges from 1 (best) to 0 (worst)."""

    def __init__(self, input_range: Tuple[int, int]=(-1, 1), device: Optional[str]=None, use_imagenet_templates: bool=False, metric: str="text_img", clip_model: str="ViT-B/16") -> None:
        """Initializes a new CLIP similarity metric.

        Args:
            input_range (Tuple[int, int], optional): Input range for image tensors needed for normalization. Defaults to (-1, 1).
            device (Optional[str], optional): Device to compute the metric on. Defaults to None.
            use_imagenet_templates (bool, optional): If True CLIP text embeddings will be averaged over 80 imagenet templates. Defaults to False.
            metric (str, optional): Which CLIP metric to compute. Available metrics are ("text_img", "img_img", "text_text", "textdir_imgdir"). Defaults to "text_img".
            clip_model (str, optional): Which CLIP model to use. Defaults to "ViT-B/16".
        """

        assert metric in ("text_img", "img_img", "text_text", "textdir_imgdir")

        super().__init__(input_range, device)
        import clip

        # load clip
        self.model, self.preprocess = clip.load(clip_model, self.device)
        self.model.eval()

        self.templates = imagenet_templates if use_imagenet_templates else ["{}"]
        self.losses = []

        self.metric = metric

        # load BLIP if needed (for text-text similarity)
        self.gen_caption = BLIPCaptionGenerator(self.device) if metric == "text_text" else None

    def get_img_feat(self, image: Image.Image) -> torch.Tensor:
        """Computes normalized CLIP image features

        Args:
            image (Image.Image): Image to compute features for.

        Returns:
            torch.Tensor: Image feature
        """

        image_preproc = self.preprocess(image).unsqueeze(0).to(self.device)

        # encode image and normalize embedding
        feat = self.model.encode_image(image_preproc)
        feat /= feat.norm(dim=-1, keepdim=True)
        feat = feat.float()
        feat = feat.squeeze(0)

        return feat
    
    def get_text_feat(self, text: str) -> torch.Tensor:
        """Computes normalized CLIP text features.

        Args:
            text (str): Text to compute features for.

        Returns:
            torch.Tensor: Text features
        """

        return get_embedding_for_prompt(self.model, text, templates=self.templates)

    def torch_to_pil(self, image: Optional[torch.Tensor]) -> Union[Image.Image, None]:
        """Converts a tensor image to a PIL image.

        Args:
            image (Optional[torch.Tensor]): Tensor to convert.

        Returns:
            Union[Image.Image, None]: PIL image
        """

        if image is None:
            return None

        image = self._normalize(image)[0].permute(1, 2, 0).detach().cpu().numpy()
        image = np.clip(image  * 255, 0, 255).astype(np.uint8)
        image = Image.fromarray(image)
        return image

    def forward(self, source_image: Optional[torch.Tensor]=None, target_image: Optional[torch.Tensor]=None, source_prompt: Optional[str]=None, target_prompt: Optional[str]=None) -> torch.Tensor:
        """Compute metric value for a single example.

        Args:
            source_image (Optional[torch.Tensor], optional): Groundtruth input image (0-1 range). Defaults to None.
            target_image (Optional[torch.Tensor], optional): Output image (0-1 range). Defaults to None.
            source_prompt (Optional[str], optional): Prompt used for inversion. Defaults to None.
            target_prompt (Optional[str], optional): Prompt used for editing. Defaults to None.

        Returns:
            torch.Tensor: Metric value (1 best, 0 worst)
        """

        source_image = self.torch_to_pil(source_image)
        target_image = self.torch_to_pil(target_image)

        if self.metric == "text_img":
            a = self.get_img_feat(target_image)
            b = self.get_text_feat(target_prompt)
        elif self.metric == "img_img":
            a = self.get_img_feat(source_image)
            b = self.get_img_feat(target_image)
        elif self.metric == "textdir_imgdir":
            a = self.get_img_feat(target_image) - self.get_img_feat(source_image)
            b = self.get_text_feat(target_prompt) - self.get_text_feat(source_prompt)
        elif self.metric == "text_text":
            pred_prompt = self.gen_caption(target_image)
            a = self.get_text_feat(pred_prompt)
            b = self.get_text_feat(target_prompt)

        # dot product
        assert a.ndim == 1 and a.shape == b.shape
        sim = torch.dot(a, b)

        return sim

    def __repr__(self) -> str:
        return f"clip_{self.metric}"


class CLIPAccuracy(SimpleMetric):
    """CLIP accuracy from pix2pix-zero (https://arxiv.org/abs/2302.03027). Ranges from 1 (best) to 0 (worst).
    Computes the CLIP image-text similarity of the output image with the source prompt and the target prompt
    and classifies an image as correctly edited if the similarity with the target prompt is higher than with the source prompt.
    The final metric is the ratio of corectly edited images by the total number of images.
    """

    def __init__(self, input_range: Tuple[int, int]=(-1, 1), device: Optional[str]=None, 
                 use_imagenet_templates: bool=False, metric: str="text_img", clip_model: str="ViT-B/16") -> None:
        """Initializes a new CLIP accuracy metric.

        Args:
            input_range (Tuple[int, int], optional): Input range for image tensors needed for normalization. Defaults to (-1, 1).
            device (Optional[str], optional): Device to compute the metric on. Defaults to None.
            use_imagenet_templates (bool, optional): If True CLIP text embeddings will be averaged over 80 imagenet templates. Defaults to False.
            metric (str, optional): Which CLIP metric to compute. Available metrics are ("text_img", "img_img", "text_text", "textdir_imgdir"). Defaults to "text_img".
            clip_model (str, optional): Which CLIP model to use. Defaults to "ViT-B/16".
        """

        super().__init__(input_range, device)

        self.clip_sim = CLIPSimilarity(input_range, device, use_imagenet_templates, metric, clip_model)

    def forward(self, source_image: Optional[torch.Tensor]=None, target_image: Optional[torch.Tensor]=None, source_prompt: Optional[str]=None, target_prompt: Optional[str]=None) -> torch.Tensor:
        """Compute metric value for a single example.

        Args:
            source_image (Optional[torch.Tensor], optional): Groundtruth input image (0-1 range). Defaults to None.
            target_image (Optional[torch.Tensor], optional): Output image (0-1 range). Defaults to None.
            source_prompt (Optional[str], optional): Prompt used for inversion. Defaults to None.
            target_prompt (Optional[str], optional): Prompt used for editing. Defaults to None.

        Returns:
            torch.Tensor: Metric value (1-best or 0-worst)
        """
        
        sim_src = self.clip_sim(target_image=target_image, source_prompt=source_prompt)
        sim_target = self.clip_sim(target_image=target_image, target_prompt=target_prompt)
        val = sim_target > sim_src

        return val.float()

    def __repr__(self) -> str:
        return f"{self.clip_sim}_acc"
