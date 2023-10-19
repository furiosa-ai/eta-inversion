import contextlib

from modules.utils.masactrl import MutualSelfAttentionControl
from modules.utils.masactrl_utils import register_attention_editor_diffusers

from .editor import Editor
from torch import Tensor
from typing import Dict, Iterator, Optional, Any
from modules.inversion.diffusion_inversion import DiffusionInversion


class MasactrlEditor(Editor):
    """MasaControl editor
    """

    def __init__(self, inverter: DiffusionInversion, no_null_source_prompt: bool=False) -> None:
        """Initiates a new editor object

        Args:
            inverter (DiffusionInversion): Inverter to use for editing
            no_null_source_prompt (bool, optional): MasaControl uses an empty source prompt "" 
            for inversion by default. True overrides this behavior. Defaults to False.
        """

        super().__init__()

        self.inverter = inverter
        self.model = self.inverter.model
        self.no_null_source_prompt = no_null_source_prompt

    @contextlib.contextmanager
    def register_editor(self) -> Iterator[None]:
        """Applies MasaControl hooks and then removes them again.
        """
        step, layer = 4, 10
        editor = MutualSelfAttentionControl(step, layer)
        register_attention_editor_diffusers(self.model, editor)
        yield
        register_attention_editor_diffusers(self.model, None)

    def edit(self, image: Tensor, source_prompt: str, target_prompt: str, cfg: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        assert cfg is None

        # create context from prompts
        src_context = self.inverter.create_context("" if not self.no_null_source_prompt else source_prompt)
        target_context = self.inverter.create_context(target_prompt)

        # diffusion inversion with the source prompt to obtain inverse latent zT
        inv_res = self.inverter.invert(image, context=src_context, guidance_scale_fwd=1)

        # apply masactrl hooks
        with self.register_editor():
            edit_res = self.inverter.sample(inv_res, context=[src_context, target_context])
        
        return {
            "image_inv": edit_res["image"][0:1],  # Image from inversion
            "image": edit_res["image"][1:2],   # Edited image
            "latent_inv": edit_res["latent"][0:1],  # z0 for inverted image
            "latent": edit_res["latent"][1:2],  # z0 for edited image
        }
