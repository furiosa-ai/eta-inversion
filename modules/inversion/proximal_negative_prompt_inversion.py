from .negative_prompt_inversion import NegativePromptInversion

import torch

import torch.nn.functional as F
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from typing import Optional, Union


def _dilate(image: torch.Tensor, kernel_size: int, stride: int=1, padding: int=0) -> torch.Tensor:
    """
    Perform dilation on a binary image using a square kernel.
    """
    # Ensure the image is binary
    assert image.max() <= 1 and image.min() >= 0
    
    # Get the maximum value in each neighborhood
    dilated_image = F.max_pool2d(image, kernel_size, stride, padding)
    
    return dilated_image


class ProximalNegativePromptInversion(NegativePromptInversion):
    """Proximal negative prompt inversion implementation. Based on https://github.com/phymhan/prompt-to-prompt
    """

    dft_prox = "l0"
    dft_quantile = 0.7
    dft_recon_lr = 1
    dft_recon_t = 400
    dft_dilate_mask = 1

    def __init__(self, model: StableDiffusionPipeline, scheduler: Optional[str]=None, num_inference_steps: Optional[int]=None, 
                 guidance_scale_bwd: Optional[float]=None, guidance_scale_fwd: Optional[float]=None,
                 verbose: bool=False, prox: str="l0", quantile: float=0.7, recon_lr: int=1, recon_t: int=400, dilate_mask: int=1) -> None:
        """Creates a new proximal negative prompt inversion instance.

        Args:
            model (StableDiffusionPipeline): The diffusion model to invert. Must be Stable Diffusion for now.
            scheduler (Optional[str], optional): Name of the scheduler to invert. 
            Possbile choices are "ddim", "dpm" and "ddpm". Defaults to "ddim".
            num_inference_steps (Optional[int], optional): Number of denoising steps. Usually set to 50. Defaults to None.
            guidance_scale_bwd (Optional[float], optional): Classifier-free guidance scale for backward process (denoising). Defaults to None.
            guidance_scale_fwd (Optional[float], optional): Classifier-free guidance scale for forward process (inversion). Defaults to None.
            verbose (bool, optional): If True, print debug messages. Defaults to False.
            prox (str, optional): Proximal guidance type. Must be "l0" or "l1. Defaults to "l0".
            quantile (float, optional): ProxNPI quantile. Defaults to 0.7.
            recon_lr (int, optional): ProxNPI recon_lr. Defaults to 1.
            recon_t (int, optional): ProxNPI recon_t. Defaults to 400.
            dilate_mask (int, optional): Mask dilation radius. Defaults to 1.
        """

        super().__init__(model, scheduler, num_inference_steps, guidance_scale_bwd, guidance_scale_fwd, verbose)

        self.prox = prox
        self.quantile = quantile
        self.recon_t = recon_t
        self.recon_lr = recon_lr
        self.dilate_mask = dilate_mask

    def proximal_guidance(self, noise_pred_uncond: torch.Tensor, noise_prediction_text: torch.Tensor, t: torch.Tensor, guidance_scale: float) -> torch.Tensor:
        """Performs proximal guidance.

        Args:
            noise_pred_uncond (torch.Tensor): Unconditional noise prediction
            noise_prediction_text (torch.Tensor): Conditional noise prediction
            t (torch.Tensor): Timestep
            guidance_scale (float): Classifier-free guidance scale

        Returns:
            torch.Tensor: Final noise prediction
        """

        step_kwargs = {
            'ref_image': None,
            'recon_lr': 0,
            'recon_mask': None,
        }
        image_enc = None
        mask_edit = None
        if self.prox is not None:
            if self.prox == 'l1':
                score_delta = noise_prediction_text - noise_pred_uncond
                if self.quantile > 0:
                    threshold = score_delta.abs().quantile(self.quantile)
                else:
                    threshold = -self.quantile  # if quantile is negative, use it as a fixed threshold
                score_delta -= score_delta.clamp(-threshold, threshold)
                score_delta = torch.where(score_delta > 0, score_delta-threshold, score_delta)
                score_delta = torch.where(score_delta < 0, score_delta+threshold, score_delta)
                if (self.recon_t > 0 and t < self.recon_t) or (self.recon_t < 0 and t > -self.recon_t):
                    step_kwargs['ref_image'] = image_enc
                    step_kwargs['recon_lr'] = self.recon_lr
                    mask_edit = (score_delta.abs() > threshold).float()
                    if self.dilate_mask > 0:
                        radius = int(self.dilate_mask)
                        mask_edit = _dilate(mask_edit.float(), kernel_size=2*radius+1, padding=radius)
                    step_kwargs['recon_mask'] = 1 - mask_edit
            elif self.prox == 'l0':
                score_delta = noise_prediction_text - noise_pred_uncond
                if self.quantile > 0:
                    if score_delta.dtype == torch.float16:
                        threshold = score_delta.abs().float().quantile(self.quantile)
                    else:
                        threshold = score_delta.abs().quantile(self.quantile)
                else:
                    threshold = -self.quantile  # if quantile is negative, use it as a fixed threshold
                score_delta -= score_delta.clamp(-threshold, threshold)
                if (self.recon_t > 0 and t < self.recon_t) or (self.recon_t < 0 and t > -self.recon_t):
                    step_kwargs['ref_image'] = image_enc
                    step_kwargs['recon_lr'] = self.recon_lr
                    mask_edit = (score_delta.abs() > threshold).float()
                    if self.dilate_mask > 0:
                        radius = int(self.dilate_mask)
                        mask_edit = _dilate(mask_edit.float(), kernel_size=2*radius+1, padding=radius)
                    step_kwargs['recon_mask'] = 1 - mask_edit
            else:
                raise NotImplementedError
            noise_pred = noise_pred_uncond + guidance_scale * score_delta
        else:
            noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

        assert step_kwargs['ref_image'] is None
        del step_kwargs['ref_image']
        del step_kwargs['recon_lr']
        del step_kwargs["recon_mask"]

        return noise_pred

    def predict_noise(self, latent: torch.Tensor, t: torch.Tensor, context: torch.Tensor, guidance_scale: Union[float, int], is_fwd: bool=False, **kwargs) -> torch.Tensor:
        if guidance_scale is None:
            noise_pred = self.unet(latent, t, encoder_hidden_states=context, **kwargs)["sample"] 
        else:
            # cfg
            
            # duplicate latent at the batch dimension to match uncond and cond embedding in context for cfg
            if latent.shape[0] * 2 == context.shape[0]:
                latent = torch.cat([latent] * 2)
            else:
                assert latent.shape[0] == context.shape[0]

            noise_pred = self.unet(latent, t, encoder_hidden_states=context, **kwargs)["sample"]
            noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)

            if not is_fwd:
                # only apply proxnpi in backward/denoise process
                noise_pred = self.proximal_guidance(noise_pred_uncond, noise_prediction_text, t, guidance_scale)
            else:
                noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        
        return noise_pred
