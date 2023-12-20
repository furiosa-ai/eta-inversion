

from .editor import Editor
from torch import Tensor
from typing import Dict, Any, Optional
from modules.inversion.diffusion_inversion import DiffusionInversion


class InversionEditor(Editor):
    """Simple editor which just performs denoising with the target prompt without any additional modifications
    """

    def __init__(self, inverter: DiffusionInversion, no_source_backward: bool=False, vae_rec: bool=False, no_null_source_prompt: bool=True) -> None:
        """Initiates a new editor object

        Args:
            inverter (DiffusionInversion): Inverter to use for editing.
            no_source_backward (bool, optional): If True, only target prompt is used for backward. Defaults to False.
            vae_rec (bool, optional): If True, no diffusion is performed, only VAE reconstruction. Defaults to False.
            no_null_source_prompt (bool, optional): If True, will use null prompt instead of source prompt for inversion. Defaults to True.
        """

        super().__init__()

        self.inverter = inverter
        self.model = self.inverter.model
        self.no_source_backward = no_source_backward
        self.vae_rec = vae_rec
        self.no_null_source_prompt = no_null_source_prompt

    def edit(self, image: Tensor, source_prompt: str, target_prompt: str, cfg: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        assert cfg is None

        if self.vae_rec:
            latent = self.inverter.encode(image)
            image_inv = self.inverter.decode(latent)

            return {
                "image": image_inv, 
                "latent": latent,
            }
        else:
            src_context = self.inverter.create_context(source_prompt if self.no_null_source_prompt else "")

            # self.inverter.guidance_scale_bwd
            inv_res = self.inverter.invert(image, context=src_context)  # , guidance_scale_fwd=1

            edit_res = self.inverter.sample(inv_res, context=[src_context])
            
            return {
                "image": edit_res["image"], 
                "latent": edit_res["latent"],
            }
