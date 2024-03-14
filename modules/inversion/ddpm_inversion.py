import torch
from .diffusion_inversion import DiffusionInversion
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from typing import Any, Dict, List, Optional, Tuple, Union

from diffusers import DDIMScheduler
from modules.inverse_schedulers import DDPMInverseScheduler


class DDPMInversion(DiffusionInversion):
    """DDPM inversion implementation. Based on https://github.com/inbarhub/DDPM_inversion
    """

    dft_skip_steps = 0.36
    dft_forward_seed = 0

    def __init__(self, model: StableDiffusionPipeline, scheduler: Optional[str]=None, num_inference_steps: Optional[int]=None, 
                 guidance_scale_bwd: Optional[float]=None, guidance_scale_fwd: Optional[float]=None,
                 verbose: bool=False, forward_seed: Optional[int]=0, skip_steps: Optional[float]=None,
                 markovian_forward: bool=False) -> None:
        """Creates a new ddpm inversion instance.

        Args:
            model (StableDiffusionPipeline): The diffusion model to invert. Must be Stable Diffusion for now.
            scheduler (Optional[str], optional): Name of the scheduler to invert. 
            Possbile choices are "ddim", "dpm" and "ddpm". Defaults to "ddim".
            num_inference_steps (Optional[int], optional): Number of denoising steps. Usually set to 50. Defaults to None.
            guidance_scale_bwd (Optional[float], optional): Classifier-free guidance scale for backward process (denoising). Defaults to None.
            guidance_scale_fwd (Optional[float], optional): Classifier-free guidance scale for forward process (inversion). Defaults to None.
            verbose (bool, optional): If True, print debug messages. Defaults to False.
            forward_seed (Optional[int], optional): Make forward process deterministic. Defaults to 0.
            skip_steps (Optional[float], optional): How many steps to skip in the reverse process. Defaults to None.
            markovian_forward (bool, optional): If True, x_t is sampled from x_t-1, otherwise x_t is sampled from x_0. Defaults to False.
        """

        scheduler = scheduler or "ddpm"
        guidance_scale_fwd = guidance_scale_fwd or 3.5
        guidance_scale_bwd = guidance_scale_bwd or 9
        self.skip_steps = skip_steps or 0.36
        self.forward_seed = forward_seed if forward_seed >= 0 else None
        self.markovian_forward = markovian_forward
        
        super().__init__(model, scheduler, num_inference_steps, guidance_scale_bwd, guidance_scale_fwd, verbose)

    def create_schedulers(self, model: StableDiffusionPipeline, scheduler: str, num_inference_steps: int) -> Tuple[DDIMScheduler, DDIMScheduler, DDPMInverseScheduler]:
        return super().create_schedulers(model, "ddpm", num_inference_steps, scheduler_inv_kwargs=dict(markovian_forward=self.markovian_forward))

    def predict_step_forward(self, latent: torch.Tensor, t: torch.Tensor, context: torch.Tensor, guidance_scale_fwd: float, xts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform one forward diffusion steps. Makes a noise prediction using SD's UNet first and then updates the latent using the noise scheduler.

        Args:
            latent (torch.Tensor): Current latent
            t (torch.Tensor): Timestep
            context (torch.Tensor): Prompt embeddings
            guidance_scale_fwd (float): Guidance scale for classifier-free guidance. Set to None for default default scale. Defaults to None.
            xts (torch.Tensor): Noised latents

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: latent, noise prediction and variance noise
        """

        guidance_scale_fwd = guidance_scale_fwd or self.guidance_scale_fwd

        noise_pred = self.predict_noise(latent, t, context, guidance_scale_fwd, is_fwd=False)
        res = self.step_forward(noise_pred, t, latent, xts)
        latent = res.prev_sample
        variance_noise = res.variance_noise

        return latent, noise_pred, variance_noise

    def diffusion_forward(self, latent: torch.Tensor, context: torch.Tensor, guidance_scale_fwd: Optional[float]=None
                          ) -> Dict[str, Any]:
        # diffusion inversion
        xts = self.scheduler_fwd.sample_latents(latent, generator=torch.Generator(self.device).manual_seed(self.forward_seed) if self.forward_seed is not None else None)

        guidance_scale_fwd = guidance_scale_fwd or self.guidance_scale_fwd

        # first input latent also gets numerically corrected later
        latents = []
        noise_preds = []
        variance_noises = []
        etas = []

        for i, t in enumerate(self.pbar(self.scheduler_fwd.timesteps, desc="forward")):
            # print(f"t = {t.item()}")

            latent = self.scheduler_fwd.get_sampled_latent_by_t(xts, t)

            # latent is numerically corrected latent
            latent, noise_pred, variance_noise = self.predict_step_forward(latent, t, context, guidance_scale_fwd, xts)

            noise_preds.append(noise_pred)
            latents.append(latent)
            variance_noises.append(variance_noise)
            etas.append(self.scheduler_fwd.get_eta_by_t(t))

        # append final inverse latent which does not get numerically corrected
        latents.append(xts[0][None])

        # set first noise map to zero
        variance_noises[0] = torch.zeros_like(variance_noises[0])
        # variance_noises[-1] = torch.zeros_like(variance_noises[-1])

        return {"latents": latents, "noise_preds": noise_preds, "etas": etas, "variance_noises": variance_noises}

    def skip_inv_result(self, inv_result: Dict[str, Any]) -> Dict[str, Any]:
        """Cut off results from inversion to skip early denoising steps

        Args:
            inv_result (Dict[str, Any]): Inversion results

        Returns:
            Dict[str, Any]: Cut off inversion result
        """

        skip = self.get_bwd_skip()
        inv_result = {k: inv_result[k][:-skip] for k in ("latents", "noise_preds", "variance_noises", "etas")}
        return inv_result

    def get_bwd_skip(self) -> int:
        """How many denoising steps to skip

        Returns:
            int: Number of steps to skip
        """
        return int(self.skip_steps * len(self.scheduler_bwd.timesteps))
    
    def sample(self, inv_result: Dict[str, Any], prompt: Optional[Union[str, List[str]]]=None, 
               context: Optional[Union[torch.Tensor, List[torch.Tensor]]]=None) -> Dict[str, Any]:
        if self.skip_steps is not None:
            # cut off inversion result to skip steps
            inv_result = self.skip_inv_result(inv_result)
        return super().sample(inv_result, prompt=prompt, context=context)

    def predict_step_backward(self, latent: torch.Tensor, t: torch.Tensor, context: torch.Tensor, eta: float, variance_noise: torch.Tensor, guidance_scale_bwd: Optional[float]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            latent (torch.Tensor): Current latent
            t (torch.Tensor): Timestep
            context (torch.Tensor): Prompt embeddings
            eta (float): Eta to use for backward step. 1 for now
            variance_noise (torch.Tensor): Noise maps from the forward process
            guidance_scale_bwd (Optional[float], optional): Guidance scale for classifier-free guidance. Set to None for default default scale. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: updated latent and noise prediction
        """

        guidance_scale_bwd = guidance_scale_bwd or self.guidance_scale_bwd

        latent = self.controller.begin_step(latent=latent)

        if latent.shape[0] == 2:
            guidance_scale = torch.tensor([self.guidance_scale_fwd, self.guidance_scale_bwd], 
                                        device=self.model.device, dtype=latent.dtype)[:, None, None, None]
        else:
            assert latent.shape[0] == 1
            guidance_scale = self.guidance_scale_bwd

        noise_pred = self.predict_noise(latent, t, context, guidance_scale)
        latent = self.step_backward(noise_pred, t, latent, eta=eta, variance_noise=variance_noise).prev_sample

        latent = self.controller.end_step(latent=latent, noise_pred=noise_pred, t=t)

        return latent, noise_pred

    def diffusion_backward(self, latent: torch.Tensor, context: torch.Tensor, inv_result: Dict[str, List[Union[torch.Tensor, int]]]) -> torch.Tensor:
        etas = list(reversed(inv_result["etas"]))
        variance_noises = list(reversed(inv_result["variance_noises"]))
        timesteps = self.scheduler_bwd.timesteps[self.get_bwd_skip():]

        # skip timesteps from bwd_skip
        for i, t in enumerate(self.pbar(timesteps, desc="backward")):
            latent, noise_pred = self.predict_step_backward(latent, t, context, etas[i], variance_noises[i], self.guidance_scale_bwd)
        
        return latent
