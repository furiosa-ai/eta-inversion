from .diffusion_inversion import DiffusionInversion

import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from typing import Dict, List, Optional, Union, Any, Tuple


class DirectInversion(DiffusionInversion):
    # """Direct inversion implementation. Based on https://github.com/phymhan/prompt-to-prompt
    # """

    def __init__(self, model: StableDiffusionPipeline, scheduler: Optional[str]=None, num_inference_steps: Optional[int]=None, 
                 guidance_scale_bwd: Optional[float]=None, guidance_scale_fwd: Optional[float]=None,
                 verbose: bool=False) -> None:
        super().__init__(model, scheduler, num_inference_steps, guidance_scale_bwd, guidance_scale_fwd, verbose)

    def predict_step_backward(self, latent: torch.Tensor, t: torch.Tensor, context: torch.Tensor, guidance_scale_bwd: Optional[float]=None, source_latent_prev: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one backward diffusion steps. Makes a noise prediction using SD's UNet first and then updates the latent using the noise scheduler.

        Args:
            latent (torch.Tensor): Current latent
            t (torch.Tensor): Timestep
            context (torch.Tensor): Prompt embeddings
            guidance_scale_bwd (Optional[float], optional): Guidance scale for classifier-free guidance. Set to None for default default scale. Defaults to None.
            source_latent_prev (Optional[torch.Tensor], optional): Source latent from inversion. Latent will be replaces by this. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: updated latent and noise prediction
        """

        guidance_scale_bwd = guidance_scale_bwd or self.guidance_scale_bwd

        # call controller callback (e.g. ptp)
        latent = self.controller.begin_step(latent=latent, t=t)

        # make a noise prediction using UNet
        noise_pred = self.predict_noise(latent, t, context, guidance_scale_bwd)

        # update the latent based on the predicted noise with the noise schedulers
        new_latent = self.step_backward(noise_pred, t, latent).prev_sample

        # direct inversion
        if source_latent_prev is not None:
            noise_loss = source_latent_prev - new_latent[:1]
            new_latent = torch.concat((new_latent[:1]+noise_loss, new_latent[1:]))
            # new_latent = torch.concat((new_latent[:1]+noise_loss[:1],new_latent[1:]))

        # call controller callback to modify latent (e.g. ptp)
        new_latent = self.controller.end_step(latent=new_latent, noise_pred=noise_pred, t=t)

        return new_latent, noise_pred

    def diffusion_backward(self, latent: torch.Tensor, context: torch.Tensor, inv_result: Dict[str, Any]) -> torch.Tensor:
        for i, t in enumerate(self.pbar(self.scheduler_bwd.timesteps, desc="backward")):
            # pass source latent
            latent, noise_pred = self.predict_step_backward(latent, t, context, source_latent_prev=inv_result["latents"][-(i+2)])
            
        return latent

    def invert(self, image: torch.Tensor, prompt: Optional[str]=None, context: Optional[torch.Tensor]=None, 
               guidance_scale_fwd: Optional[float]=None) -> Dict[str, Any]:
        fwd_result = super().invert(image, prompt, context, guidance_scale_fwd)

        return fwd_result
