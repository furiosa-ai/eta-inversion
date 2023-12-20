import contextlib
from modules.utils.pnp import register_pnp, unregister_pnp

from ..inversion.edict_inversion import EdictInversion

from .editor import Editor
from modules.inversion.diffusion_inversion import DiffusionInversion
from torch import Tensor
from typing import Dict, Iterator, List, Optional, Any


class PlugAndPlayEditor(Editor):
    """Plug-and-play editor
    """

    def __init__(self, inverter: DiffusionInversion, no_null_source_prompt: bool=False) -> None:
        """Initiates a new editor object

        Args:
            inverter (DiffusionInversion): Inverter to use for editing
            no_null_source_prompt (bool, optional): Plug-and-play uses an empty source prompt "" 
            for inversion by default. True overrides this behavior. Defaults to False.
        """

        super().__init__()

        self.inverter = inverter
        self.model = self.inverter.model
        self.negative_prompt = "ugly, blurry, black, low res, unrealistic"  # negative prompting used in pnp
        self.no_null_source_prompt = no_null_source_prompt

    @contextlib.contextmanager
    def register_editor(self) -> Iterator[None]:
        """Applies PnP hooks and then removes them again.

        Yields:
            Iterator[None]: context
        """

        register_pnp(self.model, None)
        yield
        unregister_pnp(self.model)

    def edit(self, image: Tensor, source_prompt: str, target_prompt: str, cfg: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        assert cfg is None

        # create context from prompts
        src_context = self.inverter.create_context("" if not self.no_null_source_prompt else source_prompt)
        target_context = self.inverter.create_context(target_prompt)

        # diffusion inversion with the source prompt to obtain inverse latent zT and intermediate latents
        inv_res = self.inverter.invert(image, context=src_context)

        with self.register_editor():
            if self.negative_prompt is not None and self.negative_prompt != "":
                target_context = self.inverter.create_context(target_prompt, negative_prompt=self.negative_prompt)

            edit_res = self.inverter.sample(inv_res, context=[src_context, target_context])
        
        if edit_res is None:
            return None

        return {
            "image_inv": edit_res["image"][0:1],  # Image from inversion
            "image": edit_res["image"][1:2],   # Edited image
            "latent_inv": edit_res["latent"][0:1],  # z0 for inverted image
            "latent": edit_res["latent"][1:2],  # z0 for edited image
        }
