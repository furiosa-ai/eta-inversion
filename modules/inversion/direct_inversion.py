from .diffusion_inversion import DiffusionInversion

import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from typing import Dict, List, Optional, Union, Any, Tuple


class DirectInversion(DiffusionInversion):
    # """Negative prompt inversion implementation. Based on https://github.com/phymhan/prompt-to-prompt
    # """

    def __init__(self, model: StableDiffusionPipeline, scheduler: Optional[str]=None, num_inference_steps: Optional[int]=None, 
                 guidance_scale_bwd: Optional[float]=None, guidance_scale_fwd: Optional[float]=None,
                 verbose: bool=False) -> None:
        super().__init__(model, scheduler, num_inference_steps, guidance_scale_bwd, guidance_scale_fwd, verbose)

    def predict_step_backward(self, latent: torch.Tensor, t: torch.Tensor, context: torch.Tensor, guidance_scale_bwd: Optional[float]=None, noise_loss: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one backward diffusion steps. Makes a noise prediction using SD's UNet first and then updates the latent using the noise scheduler.

        Args:
            latent (torch.Tensor): Current latent
            t (torch.Tensor): Timestep
            context (torch.Tensor): Prompt embeddings
            guidance_scale_bwd (Optional[float], optional): Guidance scale for classifier-free guidance. Set to None for default default scale. Defaults to None.
            noise_loss (Optional[torch.Tensor], optional): Direct Inversion noise_loss. Defaults to None.

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
        if noise_loss is not None:
            new_latent = torch.concat((new_latent[:1]+noise_loss[:1],new_latent[1:]))

        # call controller callback to modify latent (e.g. ptp)
        new_latent = self.controller.end_step(latent=new_latent, noise_pred=noise_pred, t=t)

        return new_latent, noise_pred

    def diffusion_backward(self, latent: torch.Tensor, context: torch.Tensor, inv_result: Dict[str, Any]) -> torch.Tensor:
        for i, t in enumerate(self.pbar(self.scheduler_bwd.timesteps, desc="backward")):
            # pass noise loss
            latent, noise_pred = self.predict_step_backward(latent, t, context, noise_loss=inv_result["noise_loss_list"][i])
            
        return latent
    
    def offset_calculate(self, latents: List[torch.Tensor], context: torch.Tensor, guidance_scale: float) -> List[torch.Tensor]:
        """Calculates Direct Inversion's noise losses

        Args:
            latents (List[torch.Tensor]): Intermediate latents from diffusion inversion
            context (torch.Tensor): Prompt embeddings
            guidance_scale (float): CFG guidance scale. Should be same as backward CFG

        Returns:
            List[torch.Tensor]: Noise loss list
        """

        noise_loss_list = []
        latent_cur = torch.concat([latents[-1]]*(context.shape[0]//2))
        for i, t in enumerate(self.scheduler_bwd.timesteps):
            latent_prev = torch.concat([latents[len(latents) - i - 2]]*latent_cur.shape[0])
            latents_prev_rec, _ = self.predict_step_backward(latent_cur, t, context, guidance_scale)
            loss = latent_prev - latents_prev_rec
                
            noise_loss_list.append(loss.detach())
            latent_cur = latents_prev_rec + loss

        return noise_loss_list

    def invert(self, image: torch.Tensor, prompt: Optional[str]=None, context: Optional[torch.Tensor]=None, 
               guidance_scale_fwd: Optional[float]=None) -> Dict[str, Any]:
        fwd_result = super().invert(image, prompt, context, guidance_scale_fwd)

        ddim_latents = fwd_result["latents"]
        noise_loss_list = self.offset_calculate(ddim_latents, context, self.guidance_scale_bwd)

        fwd_result["noise_loss_list"] = noise_loss_list

        return fwd_result
