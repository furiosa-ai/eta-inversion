from .diffusion_inversion import DiffusionInversion

from tqdm import tqdm
from torch.optim.adam import Adam
import torch
import torch.nn.functional as nnf

from diffusers import DPMSolverMultistepScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from typing import Dict, List, Optional, Union, Any


class NullTextInversion(DiffusionInversion):
    """Class implementing null-text-inversion
    """

    def __init__(self, model: StableDiffusionPipeline, scheduler: Optional[str]=None, num_inference_steps: Optional[int]=None, 
                 guidance_scale_bwd: Optional[float]=None, guidance_scale_fwd: Optional[float]=None,
                 verbose: bool=False, num_inner_steps: Optional[int]=None, early_stop_epsilon: Optional[float]=None) -> None:
        """Creates a new inverter using null-text inversion

        Args:
            model (StableDiffusionPipeline): The diffusion model to invert. Must be Stable Diffusion for now.
            scheduler (Optional[str], optional): Name of the scheduler to invert. 
            Possbile choices are "ddim", "dpm" and "ddpm". Defaults to "ddim".
            num_inference_steps (Optional[int], optional): Number of denoising steps. Usually set to 50. Defaults to None.
            guidance_scale_bwd (Optional[float], optional): Classifier-free guidance scale for backward process (denoising). Defaults to None.
            guidance_scale_fwd (Optional[float], optional): Classifier-free guidance scale for forward process (inversion). Defaults to None.
            verbose (bool, optional): If True, print debug messages. Defaults to False.
            num_inner_steps (Optional[int], optional): Number of null-text optimization steps per diffusion timestep. Defaults to None.
            early_stop_epsilon (Optional[float], optional): Early stop optimization for a timestep if loss is bellow this threshold. Defaults to None.
        """

        super().__init__(model, scheduler, num_inference_steps, guidance_scale_bwd, guidance_scale_fwd, verbose)

        self.num_inner_steps = num_inner_steps or 10
        self.early_stop_epsilon = early_stop_epsilon or 1e-5

    @torch.enable_grad()
    def null_optimization(self, latents: List[torch.Tensor], context: torch.Tensor, num_inner_steps: int, epsilon: float) -> List[torch.Tensor]:
        """Performs null-text-optimization of unconditional embeddings for accurate inversion. 
        Based on https://github.com/google/prompt-to-prompt/blob/main/null_text_w_ptp.ipynb

        Args:
            latents (List[torch.Tensor]): Intermediate latents from forward diffusion process
            context (torch.Tensor): Prompt embedding (with unconditional embeddings)
            num_inner_steps (int): Number of null-text optimization steps per diffusion timestep.
            epsilon (float): Early stop optimization for a timestep if loss is bellow this threshold.

        Returns:
            List[torch.Tensor]: Optimized unconditional embeddings
        """

        uncond_embeddings, cond_embeddings = context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=num_inner_steps * self.num_inference_steps)
        for i in range(self.num_inference_steps):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.scheduler_bwd.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.predict_noise(latent_cur, t, cond_embeddings, guidance_scale=None)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.predict_noise(latent_cur, t, uncond_embeddings, guidance_scale=None)
                noise_pred = noise_pred_uncond + self.guidance_scale_bwd * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.step_backward(noise_pred, t, latent_cur).prev_sample
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()

                if isinstance(self.scheduler_bwd, DPMSolverMultistepScheduler):
                    # for higher order solvers
                    loss.backward(retain_graph=True)
                else:
                    loss.backward()
                
                optimizer.step()
                loss_item = loss.item()
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur, _ = self.predict_step_backward(latent_cur, t, context)
        bar.close()
        return uncond_embeddings_list

    def diffusion_backward(self, latent: torch.Tensor, context: torch.Tensor, inv_result: Dict[str, Union[List[torch.Tensor], torch.Tensor]]) -> torch.Tensor:
        for i, t in enumerate(self.pbar(self.scheduler_bwd.timesteps, desc="backward")):
            context[:context.shape[0] // 2] = inv_result["uncond_embeddings"][i]  # patch in result from nti (necessary for src and target latent)
            latent, noise_pred = self.predict_step_backward(latent, t, context)
            
        return latent
    
    def invert(self, image: torch.Tensor, prompt: Optional[str]=None, context: Optional[torch.Tensor]=None, 
               guidance_scale_fwd: Optional[float]=None) -> Dict[str, Any]:
        # invert image for intermediate latents
        fwd_result = super().invert(image, prompt, context, guidance_scale_fwd)

        # perform nti
        fwd_result["uncond_embeddings"] = self.null_optimization(
            fwd_result["latents"], fwd_result["context"], self.num_inner_steps, self.early_stop_epsilon)
        return fwd_result
