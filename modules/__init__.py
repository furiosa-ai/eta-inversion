

from .inversion.diffusion_inversion import DiffusionInversion
from .inversion.null_text_inversion import NullTextInversion
from .inversion.negative_prompt_inversion import NegativePromptInversion
from .inversion.proximal_negative_prompt_inversion import ProximalNegativePromptInversion
from .inversion.edict_inversion import EdictInversion
from .inversion.ddpm_inversion import DDPMInversion

from .editing.simple_editor import SimpleEditor
from .editing.ptp_editor import PromptToPromptEditor
from .editing.masactrl_editor import MasactrlEditor
from .editing.pnp_editor import PlugAndPlayEditor
from .editing.pix2pix_zero import Pix2PixZeroEditor

from .models import StablePreprocess, StablePostProc, load_diffusion_model
from modules.editing.editor import Editor
from modules.editing.masactrl_editor import MasactrlEditor
from modules.editing.pnp_editor import PlugAndPlayEditor
from modules.editing.ptp_editor import PromptToPromptEditor
from modules.editing.simple_editor import SimpleEditor
from modules.inversion.diffusion_inversion import DiffusionInversion
from typing import Union, List, Callable


_inverters = {
    "diffinv": DiffusionInversion,
    "nti": NullTextInversion,
    "npi": NegativePromptInversion,
    "proxnpi": ProximalNegativePromptInversion,
    "edict": EdictInversion,
    "ddpminv": DDPMInversion,
}


_editors = {
    "simple": SimpleEditor,
    "ptp": PromptToPromptEditor,
    "masactrl": MasactrlEditor,
    "pnp": PlugAndPlayEditor,
    "pix2pix_zero": Pix2PixZeroEditor,
}


def register_editor(name: str, editor_cls: Callable) -> None:
    """Register a new editor for load_editor().

    Args:
        name (str): Name for the editor
        editor_cls (Callable): Editor class
    """

    print(f"Registering editor {name}")
    _editors[name] = editor_cls


def get_inversion_methods() -> List[str]:
    """Get all list of all supported inversion method names.

    Returns:
        List[str]: List of inversion method names
    """
    return list(_inverters.keys())


def get_edit_methods() -> List[str]:
    """Get all list of all supported editing method names.

    Returns:
        List[str]: List of editing method names
    """

    return list(_editors.keys())


def load_inverter(type: str, **kwargs) -> DiffusionInversion:
    """Load inverter by name.

    Args:
        type (str): inverter name

    Returns:
        DiffusionInversion: Inverter instance
    """

    return _inverters[type](**kwargs)


def load_editor(type: str, **kwargs) -> Editor:
    """Load editor by name.

    Args:
        type (str): editor name

    Returns:
        DiffusionInversion: Editor instance
    """

    return _editors[type](**kwargs)
