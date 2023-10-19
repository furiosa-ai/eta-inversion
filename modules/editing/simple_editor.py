

from .editor import Editor
from torch import Tensor
from typing import Dict, Any, Optional
from modules.inversion.diffusion_inversion import DiffusionInversion


class SimpleEditor(Editor):
    """Simple editor which just performs denoising with the target prompt without any additional modifications
    """

    def __init__(self, inverter: DiffusionInversion, no_source_backward: bool=False) -> None:
        """Initiates a new editor object

        Args:
            inverter (DiffusionInversion): Inverter to use for editing.
            no_source_backward (bool, optional): If True, only target prompt is used for backward. Defaults to False.
        """

        super().__init__()

        self.inverter = inverter
        self.model = self.inverter.model
        self.no_source_backward = no_source_backward

    def edit(self, image: Tensor, source_prompt: str, target_prompt: str, cfg: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        assert cfg is None

        src_context = self.inverter.create_context(source_prompt)
        target_context = self.inverter.create_context(target_prompt)

        # self.inverter.guidance_scale_bwd
        inv_res = self.inverter.invert(image, context=src_context, guidance_scale_fwd=1)

        if not self.no_source_backward:
            edit_res = self.inverter.sample(inv_res, context=[src_context, target_context])
            
            return {
                "image_inv": edit_res["image"][0:1], 
                "image": edit_res["image"][1:2], 
                "latent_inv": edit_res["latent"][0:1],
                "latent": edit_res["latent"][1:2],
            }
        else:
            edit_res = self.inverter.sample(inv_res, context=[target_context])
            
            return {
                "image": edit_res["image"], 
                "latent": edit_res["latent"],
            }
