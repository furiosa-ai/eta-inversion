from .diffusion_inversion import DiffusionInversion

import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from typing import Dict, List, Optional, Union, Any


class NegativePromptInversion(DiffusionInversion):
    """Negative prompt inversion implementation. Based on https://github.com/phymhan/prompt-to-prompt
    """

    def __init__(self, model: StableDiffusionPipeline, scheduler: Optional[str]=None, num_inference_steps: Optional[int]=None, 
                 guidance_scale_bwd: Optional[float]=None, guidance_scale_fwd: Optional[float]=None,
                 verbose: bool=False) -> None:
        super().__init__(model, scheduler, num_inference_steps, guidance_scale_bwd, guidance_scale_fwd, verbose)

    def diffusion_backward(self, latent: torch.Tensor, context: torch.Tensor, inv_result: Dict[str, Any]) -> torch.Tensor:
        for i, t in enumerate(self.pbar(self.scheduler_bwd.timesteps, desc="backward")):
            # patch in conditional embeddings for null embeddings (necessary for src and target)
            context[:context.shape[0] // 2] = inv_result["uncond_embeddings"][i]  
            latent, noise_pred = self.predict_step_backward(latent, t, context)
            
        return latent
    
    def invert(self, image: torch.Tensor, prompt: Optional[str]=None, context: Optional[torch.Tensor]=None, 
               guidance_scale_fwd: Optional[float]=None, inv_cfg=None) -> Dict[str, Any]:
        fwd_result = super().invert(image, prompt, context, guidance_scale_fwd)

        uncond_embeddings, cond_embeddings = fwd_result["context"].chunk(2)  # use conditional embeddings as null embeddings
        fwd_result["uncond_embeddings"] = [cond_embeddings] * self.num_inference_steps
        return fwd_result
