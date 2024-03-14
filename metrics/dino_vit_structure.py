# Refer to https://github.com/omerbt/Splice/blob/master/util/losses.py and 
# https://github.com/omerbt/Splice/blob/master/models/extractor.py

from .base import SimpleMetric
import torch
from torchvision.transforms import Resize
from torchvision import transforms
import torch.nn.functional as F


import torch
from typing import Callable, List, Tuple, Optional


def attn_cosine_sim(x: torch.Tensor, eps: float=1e-08) -> torch.Tensor:
    x = x[0]  # TEMP: getting rid of redundant dimension, TBF
    norm1 = x.norm(dim=2, keepdim=True)
    factor = torch.clamp(norm1 @ norm1.permute(0, 2, 1), min=eps)
    sim_matrix = (x @ x.permute(0, 2, 1)) / factor
    return sim_matrix


class VitExtractor:
    BLOCK_KEY = 'block'
    ATTN_KEY = 'attn'
    PATCH_IMD_KEY = 'patch_imd'
    QKV_KEY = 'qkv'
    KEY_LIST = [BLOCK_KEY, ATTN_KEY, PATCH_IMD_KEY, QKV_KEY]

    def __init__(self, model_name: str, device: str) -> None:
        # self.model = torch.hub.load('facebookresearch/dino:main', model_name).to(device)
        self.model = torch.hub.load(
            'facebookresearch/dino:main' if not model_name.startswith("dinov2") 
            else 'facebookresearch/dinov2', model_name).to(device)
        self.model.eval()
        self.model_name = model_name
        self.hook_handlers = []
        self.layers_dict = {}
        self.outputs_dict = {}
        for key in VitExtractor.KEY_LIST:
            self.layers_dict[key] = []
            self.outputs_dict[key] = []
        self._init_hooks_data()

    def _init_hooks_data(self) -> None:
        self.layers_dict[VitExtractor.BLOCK_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.ATTN_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.QKV_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.PATCH_IMD_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        for key in VitExtractor.KEY_LIST:
            # self.layers_dict[key] = kwargs[key] if key in kwargs.keys() else []
            self.outputs_dict[key] = []

    def _register_hooks(self, **kwargs) -> None:
        assert len(self.model.blocks) == 12
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in self.layers_dict[VitExtractor.BLOCK_KEY]:
                self.hook_handlers.append(block.register_forward_hook(self._get_block_hook()))
            if block_idx in self.layers_dict[VitExtractor.ATTN_KEY]:
                self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_attn_hook()))
            if block_idx in self.layers_dict[VitExtractor.QKV_KEY]:
                self.hook_handlers.append(block.attn.qkv.register_forward_hook(self._get_qkv_hook()))
            if block_idx in self.layers_dict[VitExtractor.PATCH_IMD_KEY]:
                self.hook_handlers.append(block.attn.register_forward_hook(self._get_patch_imd_hook()))

    def _clear_hooks(self) -> None:
        for handler in self.hook_handlers:
            handler.remove()
        self.hook_handlers = []

    def _get_block_hook(self) -> Callable:
        def _get_block_output(model, input, output):
            self.outputs_dict[VitExtractor.BLOCK_KEY].append(output)

        return _get_block_output

    def _get_attn_hook(self) -> Callable:
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.ATTN_KEY].append(output)

        return _get_attn_output

    def _get_qkv_hook(self) -> Callable:
        def _get_qkv_output(model, inp, output):
            self.outputs_dict[VitExtractor.QKV_KEY].append(output)

        return _get_qkv_output

    # TODO: CHECK ATTN OUTPUT TUPLE
    def _get_patch_imd_hook(self) -> Callable:
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.PATCH_IMD_KEY].append(output[0])

        return _get_attn_output

    def get_feature_from_input(self, input_img):  # List([B, N, D])
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.BLOCK_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_qkv_feature_from_input(self, input_img: torch.Tensor) -> List[    torch.Tensor]:
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.QKV_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_attn_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.ATTN_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_patch_size(self) -> int:
        if self.model_name.startswith("dinov2"):
            return 14
        else:
            return 8 if "8" in self.model_name else 16

    def get_width_patch_num(self, input_img_shape: torch.Size) -> int:
        b, c, h, w = input_img_shape
        patch_size = self.get_patch_size()
        return w // patch_size

    def get_height_patch_num(self, input_img_shape: torch.Size) -> int:
        b, c, h, w = input_img_shape
        patch_size = self.get_patch_size()
        return h // patch_size

    def get_patch_num(self, input_img_shape: torch.Size) -> int:
        patch_num = 1 + (self.get_height_patch_num(input_img_shape) * self.get_width_patch_num(input_img_shape))
        return patch_num

    def get_head_num(self) -> int:
        if self.model_name.startswith("dinov2"):
            return {
                "dinov2_vits14": 6,
                "dinov2_vitb14": 12,
                "dinov2_vitl14": 16,
                "dinov2_vitg14": 24,
            }[self.model_name]
        else:
            if "dino" in self.model_name:
                return 6 if "s" in self.model_name else 12
            return 6 if "small" in self.model_name else 12

    def get_embedding_dim(self) -> int:
        if self.model_name.startswith("dinov2"):
            return {
                "dinov2_vits14": 384,
                "dinov2_vitb14": 768,
                "dinov2_vitl14": 1024,
                "dinov2_vitg14": 1536,
            }[self.model_name]
        else:
            if "dino" in self.model_name:
                return 384 if "s" in self.model_name else 768
            return 384 if "small" in self.model_name else 768

    # def get_dim(qkv):
    #     (batch_size, patch_num, embedding_dim3) = qkv.shape
    #     assert batch_size == 1
    #     assert embedding_dim3 % 3 == 0
    #     embedding_dim = embedding_dim3 // 3
    #     return 

    def get_queries_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()

        assert qkv.shape == (1, patch_num, embedding_dim * 3)
        q = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[0]
        return q

    def get_keys_from_qkv(self, qkv: torch.Tensor, input_img_shape: torch.Size) -> torch.Tensor:
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()

        assert qkv.shape == (1, patch_num, embedding_dim * 3)
        k = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[1]
        return k

    def get_values_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()

        assert qkv.shape == (1, patch_num, embedding_dim * 3)
        v = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[2]
        return v

    def get_keys_from_input(self, input_img: torch.Tensor, layer_num: int) -> torch.Tensor:
        qkv_features = self.get_qkv_feature_from_input(input_img)[layer_num]
        keys = self.get_keys_from_qkv(qkv_features, input_img.shape)
        return keys

    def get_keys_self_sim_from_input(self, input_img: torch.Tensor, layer_num: int) -> torch.Tensor:
        keys = self.get_keys_from_input(input_img, layer_num=layer_num)
        h, t, d = keys.shape
        concatenated_keys = keys.transpose(0, 1).reshape(t, h * d)
        ssim_map = attn_cosine_sim(concatenated_keys[None, None, ...])
        return ssim_map


class DinoVitStructure(SimpleMetric):
    """Measure structural similarity of two images using DINO features. Lower means better. Refer to https://github.com/omerbt/Splice/tree/master
    """

    def __init__(self, input_range: Tuple[int, int]=(-1, 1), device: Optional[str]=None, vit_model: Optional[str]="dino_vitb8") -> None:
        """Initializes a new metric object.

        Args:
            input_range (Tuple[int, int], optional): Input range for image tensors needed for normalization. Defaults to (-1, 1).
            device (Optional[str], optional): Device to compute the metric on. Defaults to None.
            vit_model (Optional[str], optional): DINO model to use for feature extraction. Defaults to "dino_vitb8".
        """

        super().__init__(input_range, device)

        self.vit_model = vit_model
        cfg = {
            'dino_model_name': vit_model,  # ['dino_vitb8', 'dino_vits8', 'dino_vitb16', 'dino_vits16']
            'dino_global_patch_size': 224,
        }

        self.extractor = VitExtractor(model_name=cfg['dino_model_name'], device=self.device)

        imagenet_norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        global_resize_transform = Resize(cfg['dino_global_patch_size'], max_size=480)

        self.global_transform = transforms.Compose([global_resize_transform,
                                                    imagenet_norm
                                                    ])

        self.losses = []

    def calculate_global_ssim_loss(self, outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """Computes global ssim. Adapted from https://github.com/omerbt/Splice/blob/master/util/losses.py

        Args:
            outputs (torch.Tensor): Output images
            inputs (torch.Tensor): Input images

        Returns:
            torch.Tensor: Loss value
        """

        loss = 0.0
        for a, b in zip(inputs, outputs):  # avoid memory limitations
            a = self.global_transform(a)
            b = self.global_transform(b)
            target_keys_self_sim = self.extractor.get_keys_self_sim_from_input(a.unsqueeze(0), layer_num=11)
            keys_ssim = self.extractor.get_keys_self_sim_from_input(b.unsqueeze(0), layer_num=11)
            loss += F.mse_loss(keys_ssim, target_keys_self_sim)
        return loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute metric value for a single example.

        Args:
            pred (torch.Tensor): Output image
            target (torch.Tensor): Groundtruth image

        Returns:
            torch.Tensor: Metric value
        """

        pred = self._normalize(pred)
        target = self._normalize(target)

        loss = self.calculate_global_ssim_loss(pred, target)

        return loss

    def __repr__(self) -> str:
        return self.vit_model
