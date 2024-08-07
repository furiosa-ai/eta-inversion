

import torch
from typing import Callable, Dict, List, Union, Any, Optional
from modules.editing.controller import ControllerBase
from modules.inversion.diffusion_inversion import DiffusionInversion

class Editor:
    """Base class for all editors
    """

    def __init__(self) -> None:
        """Initiates a new editor object
        """
        pass

    def edit(self, image: torch.Tensor, source_prompt: str, target_prompt: str, cfg: Optional[Dict[str, Any]]=None, **kwargs) -> Dict[str, Any]:
        """Edit the image

        Args:
            image (torch.Tensor): source image to edit
            source_prompt (str): source prompt for inversion
            target_prompt (str): target prompt for editing
            cfg (Optional[Dict[str, Any]], optional): Additional configuration (needed for e.g., prompt-to-prompt). Defaults to None.

        Returns:
            Dict[str, Any]: Editing result
        """
        raise NotImplementedError


class ControllerBasedEditor(Editor):
    """Base editor using a controller
    """

    def __init__(self, inverter: DiffusionInversion, no_source_backward: bool=False, dft_cfg: Optional[Dict[Any, str]]=None, fake_edit: bool=False) -> None:
        """Initiates a new editor object

        Args:
            inverter (DiffusionInversion): Inverter to use for editing.
            no_source_backward (bool, optional): If True, only target prompt is used for backward. Defaults to False.
            dft_cfg (Optional[Dict[Any, str]], optional): Default config to use for editing. Used if no config is provided to edit(). Defaults to None.
            fake_edit (bool, optional): If true editing will ignore input image and use the provided latent for editing. Defaults to False.
        """

        super().__init__()

        self.inverter = inverter
        self.no_source_backward = no_source_backward
        self.dft_cfg = dft_cfg if dft_cfg is not None else {}
        self.fake_edit = fake_edit

    def make_controller(self, image: torch.Tensor, source_prompt: str, target_prompt: str, inv_res: Dict[str, Any], **kwargs) -> ControllerBase:
        """Creates a new controller for editing the current image

        Args:
            image (torch.Tensor): Input image tensor.
            source_prompt (str): Source prompt.
            target_prompt (str): Target prompt.
            inv_res (Dict[str, Any]): Result from inversion.

        Returns:
            ControllerBase: New controller
        """
        raise NotImplementedError

    def edit(self, image: torch.Tensor, source_prompt: str, target_prompt: str, cfg: Optional[Dict[str, Any]]=None, inv_cfg=None, **kwargs) -> Dict[str, Any]:
        if cfg is None:
            cfg = {**self.dft_cfg}

        if inv_cfg is None:
            inv_cfg = {}

        # self.inverter.guidance_scale_fwd = 2.0
        # self.inverter.guidance_scale_bwd = 7.5

        # # create context from prompts
        # src_context = self.inverter.create_context(source_prompt, negative_prompt=target_prompt)
        # target_context = self.inverter.create_context(target_prompt, negative_prompt=source_prompt)
        
        src_context = self.inverter.create_context(source_prompt)
        target_context = self.inverter.create_context(target_prompt)

        # diffusion inversion with the source prompt to obtain inverse latent zT

        zT_gt = cfg.pop("zT_gt", None)
        if self.fake_edit:
            image = None
            # no inversion needed if fake editing
            inv_res = {"latents": [zT_gt.to(self.inverter.model.device)]}
        else:
            inv_res = self.inverter.invert(image, prompt=source_prompt, context=src_context, inv_cfg=inv_cfg)  # , guidance_scale_fwd=1

        # prepare controller
        controller = self.make_controller(image=image, source_prompt=source_prompt, target_prompt=target_prompt, inv_res=inv_res, **cfg, **kwargs)

        # sample a new image from the inverse latent zT
        # diffusion step is modified by the controller
        with self.inverter.use_controller(controller):
            if not self.no_source_backward:
                edit_res = self.inverter.sample(inv_res, context=[src_context, target_context])
                
                if edit_res is None:
                    return None

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


class ControllerBasedEditorLambda(ControllerBasedEditor):
    def __init__(self, inverter: DiffusionInversion, controller_cls: Optional[Callable]=None, no_source_backward: bool=False, **kwargs) -> None:
        """Initiates a new editor object

        Args:
            inverter (DiffusionInversion): Inverter to use for editing
            controller_cls (Optional[Callable], optional): Controller class to use for editing. Defaults to None.
        """

        super().__init__(inverter, no_source_backward=no_source_backward)

        self.controller_cls = controller_cls
        self.controller_kwargs = kwargs

    def make_controller(self, image: torch.Tensor, source_prompt: str, target_prompt: str, **kwargs) -> ControllerBase:
        return self.controller_cls(editor=self, image=image, source_prompt=source_prompt, target_prompt=target_prompt, **kwargs, **self.controller_kwargs)