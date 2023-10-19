
from ..editing.controller import EdictController, ControllerEmpty
from .diffusion_inversion import DiffusionInversion
import contextlib

import torch
from diffusers import DDIMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput
from ..editing.controller import ControllerBase, ControllerEmpty
import diffusers.schedulers.scheduling_ddim
from diffusers.configuration_utils import FrozenDict
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from modules.editing.ptp_editor import PromptToPromptController
from typing import Iterator, List, Optional, Tuple


class EdictSchedulerBase:
    """Scheduler wrapper to use for Edict inversion.
    """

    def __init__(self, scheduler: Optional[DDIMScheduler]=None) -> None:
        """Creates a new Edict scheduler for Edict from the passed scheduler.

        Args:
            scheduler (Optional[DDIMScheduler], optional): Scheduler to wrap. Uses DDIM if not passed. Defaults to None.
        """

        if scheduler is None:
            scheduler = DDIMScheduler(
                beta_start=0.00085, beta_end=0.012,
                beta_schedule='scaled_linear',
                num_train_timesteps=1000,
                clip_sample=False,
                set_alpha_to_one=False)

        self.scheduler = scheduler

    def set_timesteps(self, num_inference_steps: int) -> None:
        """Set the number of inference steps

        Args:
            num_inference_steps (int): Number of inverse steps
        """
        self.scheduler.set_timesteps(num_inference_steps)

    @property
    def config(self) -> FrozenDict:
        return self.scheduler.config

    @property
    def timesteps(self) -> torch.Tensor:
        return self.scheduler.timesteps

    @property
    def num_inference_steps(self) -> int:
        return self.scheduler.num_inference_steps
    
    @property
    def alphas_cumprod(self) -> torch.Tensor:
        return self.scheduler.alphas_cumprod

    def get_alpha_and_beta(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get noise scheduler alpha and beta at timestep t.

        Args:
            t (torch.Tensor): Timestep

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Alpha and beta
        """

        scheduler = self.scheduler

        # want to run this for both current and previous timnestep
        if t.dtype==torch.long:
            alpha = scheduler.alphas_cumprod[t]
            return alpha, 1-alpha
        
        if t<0:
            return scheduler.final_alpha_cumprod, 1 - scheduler.final_alpha_cumprod

        
        low = t.floor().long()
        high = t.ceil().long()
        rem = t - low
        
        low_alpha = scheduler.alphas_cumprod[low]
        high_alpha = scheduler.alphas_cumprod[high]
        interpolated_alpha = low_alpha * rem + high_alpha * (1-rem)
        interpolated_beta = 1 - interpolated_alpha
        return interpolated_alpha, interpolated_beta
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor
    ) -> DDIMSchedulerOutput:
        """Perform a scheduler step and update the current latent

        Args:
            model_output (torch.Tensor):  Noise prediction from the diffusion model
            timestep (int): Timestep
            sample (torch.Tensor): Current latent

        Returns:
            DDIMSchedulerOutput: Updated latent result
        """

        raise NotImplementedError


class EdictScheduler(EdictSchedulerBase):
    """Edict reverse (denoise) scheduler.
    """

    def __init__(self, scheduler: Optional[DDIMScheduler]=None) -> None:
        super().__init__(scheduler=scheduler)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor
    ) -> DDIMSchedulerOutput:
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        prev_timestep = timestep - self.config.num_train_timesteps / self.num_inference_steps
            
        if timestep > self.timesteps.max():
            raise NotImplementedError("Need to double check what the overflow is")

        alpha_prod_t, beta_prod_t = self.get_alpha_and_beta(timestep)
        alpha_prod_t_prev, _ = self.get_alpha_and_beta(prev_timestep)


        alpha_quotient = ((alpha_prod_t / alpha_prod_t_prev)**0.5)
        first_term =  (1./alpha_quotient) * sample
        second_term = (1./alpha_quotient) * (beta_prod_t ** 0.5) * model_output
        third_term = ((1 - alpha_prod_t_prev)**0.5) * model_output
        return DDIMSchedulerOutput(first_term - second_term + third_term)


class EdictSchedulerInverse(EdictSchedulerBase):
    """Edict forward (inverse) scheduler.
    """

    def __init__(self, scheduler: Optional[DDIMScheduler]=None) -> None:
        super().__init__(scheduler=scheduler)

    @property
    def timesteps(self) -> torch.Tensor:
        return self.scheduler.timesteps.flip(0)
        # return list(reversed(self.scheduler.timesteps))

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor
    ) -> DDIMSchedulerOutput:
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        prev_timestep = timestep - self.config.num_train_timesteps / self.num_inference_steps

        if timestep > self.timesteps.max():
            raise NotImplementedError
        else:
            alpha_prod_t = self.alphas_cumprod[timestep]
            
        alpha_prod_t, beta_prod_t = self.get_alpha_and_beta(timestep)
        alpha_prod_t_prev, _ = self.get_alpha_and_beta(prev_timestep)
        
        alpha_quotient = ((alpha_prod_t / alpha_prod_t_prev)**0.5)
        
        first_term =  alpha_quotient * sample
        second_term = ((beta_prod_t)**0.5) * model_output
        third_term = alpha_quotient * ((1 - alpha_prod_t_prev)**0.5) * model_output
        return DDIMSchedulerOutput(first_term + second_term - third_term)


class EdictInversion(DiffusionInversion):
    def __init__(self, model: StableDiffusionPipeline, scheduler: Optional[str]=None, num_inference_steps: Optional[int]=None, 
                 guidance_scale_bwd: Optional[float]=None, guidance_scale_fwd: Optional[float]=None,
                 verbose: bool=False, mix_weight: float=0.93, leapfrog_steps: bool=True, init_image_strength: float=0.8) -> None:
        """Creates a new edict inversion instance

        Args:
            model (StableDiffusionPipeline): The diffusion model to invert. Must be Stable Diffusion for now.
            scheduler (Optional[str], optional): Name of the scheduler to invert. 
            Possbile choices are "ddim", "dpm" and "ddpm". Defaults to "ddim".
            num_inference_steps (Optional[int], optional): Number of denoising steps. Usually set to 50. Defaults to None.
            guidance_scale_bwd (Optional[float], optional): Classifier-free guidance scale for backward process (denoising). Defaults to None.
            guidance_scale_fwd (Optional[float], optional): Classifier-free guidance scale for forward process (inversion). Defaults to None.
            verbose (bool, optional): If True, print debug messages. Defaults to False.
            mix_weight (float, optional): Mixing strength for latent pair syncing. Defaults to 0.93.
            leapfrog_steps (bool, optional): Edict leapfrog steps if True. Defaults to True.
            init_image_strength (float, optional): How many steps to skip at the end of inversion and the start of denoising. Defaults to 0.8.
        """
        
        guidance_scale_fwd = guidance_scale_fwd or 3.0
        guidance_scale_bwd = guidance_scale_bwd or 3.0

        super().__init__(model, scheduler, num_inference_steps, guidance_scale_bwd, guidance_scale_fwd, verbose)

        self.mix_weight = mix_weight
        self.leapfrog_steps = leapfrog_steps
        self.init_image_strength = init_image_strength
        self.t_limit = self.num_inference_steps - int(self.num_inference_steps * init_image_strength)

        with self.use_controller(None):
            pass

    @contextlib.contextmanager
    def use_controller(self, controller: Optional[ControllerBase]) -> Iterator[None]:
        if controller is None:
            controller = ControllerEmpty()

        # wrap controller
        self.controller = EdictController(controller)
        self.controller.begin()
        yield
        self.controller.end()
        self.controller = EdictController(ControllerEmpty())

    def create_schedulers(self, model: StableDiffusionPipeline, scheduler: str, num_inference_steps: int) -> Tuple[DDIMScheduler, EdictScheduler, EdictSchedulerInverse]:
        scheduler, scheduler_bwd, scheduler_fwd = super().create_schedulers(model, scheduler, num_inference_steps)

        # wrap schedulers
        scheduler_bwd_new = EdictScheduler(scheduler_bwd)
        scheduler_fwd_new = EdictSchedulerInverse(scheduler_bwd)  # discard scheduler_fwd

        return scheduler, scheduler_bwd_new, scheduler_fwd_new

    def iter_latent_pair(self, i: int, latent_pair: List[torch.Tensor], is_fwd: bool=False) -> Iterator[Tuple[int, Tuple[torch.Tensor, torch.Tensor]]]:
        """Iterate over the latent pair for the current timesteps in edict's predefined order. One iteration per latent.

        Args:
            i (int): Step index
            latent_pair (List[torch.Tensor]): Current latent pair.
            is_fwd (bool, optional): True if in forward diffusion process. Defaults to False.

        Yields:
            Iterator[Tuple[int, Tuple[torch.Tensor, torch.Tensor]]]: Selected index (0 or 1), selected latent, other latent
        """

        for latent_i in range(2):
            if is_fwd:
                if self.leapfrog_steps:
                    # what i would be from going other way
                    orig_i = len(self.scheduler_fwd.timesteps) - (i+1) 
                    offset = (orig_i+1) % 2
                    latent_i = (latent_i + offset) % 2
                else:
                    # Do 1 then 0
                    latent_i = (latent_i+1)%2
            else:
                offset = i%2
                latent_i = (latent_i + offset) % 2

            latent_j = ((latent_i+1) % 2)
            yield (latent_i, (latent_pair[latent_i], latent_pair[latent_j]))

    def sync_latent_pair(self, latent_pair: List[torch.Tensor], is_fwd: bool) -> List[torch.Tensor]:
        """Synchronizes the latent pair to avoid divergence.

        Args:
            latent_pair (List[torch.Tensor]): Latent pair to synchronize
            is_fwd (bool): True if in forward diffusion process.

        Returns:
            List[torch.Tensor]: Synchronizes latent pair.
        """

        new_latents = [l.clone() for l in latent_pair]
        if is_fwd:
            # forward process
            new_latents[1] = (new_latents[1].clone() - (1-self.mix_weight)*new_latents[0].clone()) / self.mix_weight
            new_latents[0] = (new_latents[0].clone() - (1-self.mix_weight)*new_latents[1].clone()) / self.mix_weight
        else:
            # reverse process
            new_latents[0] = (self.mix_weight*new_latents[0] + (1-self.mix_weight)*new_latents[1]).clone() 
            new_latents[1] = ((1-self.mix_weight)*new_latents[0] + (self.mix_weight)*new_latents[1]).clone() 
        latent_pair = new_latents
        return new_latents

    def step_forward_single(self, latent_idx: int, latent_base: torch.Tensor, latent_model_input: torch.Tensor, 
                            t: torch.Tensor, context: torch.Tensor, guidance_scale: float) -> torch.Tensor:
        """Perform a single forward step (noise prediction and scheduler step) for one latent in the latent pair

        Args:
            latent_idx (int): index (0 or 1) of selected latent to update
            latent_base (torch.Tensor): Selected latent to update
            latent_model_input (torch.Tensor): Other latent
            t (torch.Tensor): Current timestep
            context (torch.Tensor): Embedded prompt(s)
            guidance_scale (float): Classifier-free guidance scale

        Returns:
            torch.Tensor: Updated selected latent
        """

        noise_pred = self.predict_noise(latent_model_input, t, context, guidance_scale, is_fwd=True)
        new_latent = self.scheduler_fwd.step(noise_pred, t, latent_base).prev_sample
        return new_latent.to(latent_base.dtype)
    
    def step_backward_single(self, latent_idx: int, latent_base: torch.Tensor, latent_model_input: torch.Tensor, 
                             t: torch.Tensor, context: torch.Tensor, guidance_scale: float) -> torch.Tensor:
        """Perform a single backward step (noise prediction and scheduler step) for one latent in the latent pair

        Args:
            latent_idx (int): index (0 or 1) of selected latent to update
            latent_base (torch.Tensor): Selected latent to update
            latent_model_input (torch.Tensor): Other latent
            t (torch.Tensor): Current timestep
            context (torch.Tensor): Embedded prompt(s)
            guidance_scale (float): Classifier-free guidance scale

        Returns:
            torch.Tensor: Updated selected latent
        """

        # controller callback
        self.controller.begin_step(latent_idx)

        noise_pred = self.predict_noise(latent_model_input, t, context, guidance_scale, is_fwd=False)
        new_latent = self.scheduler_bwd.step(noise_pred, t, latent_base).prev_sample
        new_latent = new_latent.to(latent_base.dtype)
        
        # controller callback
        new_latent = self.controller.end_step(latent=new_latent)
        return new_latent

    def step_forward(self, latent: List[torch.Tensor], t: torch.Tensor, context: torch.Tensor, guidance_scale_fwd: Optional[int]=None) -> Tuple[List[torch.Tensor], None]:
        latent_pair = latent
        guidance_scale_fwd = guidance_scale_fwd or self.guidance_scale_fwd
        i = self.fwd_t_to_i[t.item()]

        # synchronize latent pair to avoid divergance of latent pair
        latent_pair = self.sync_latent_pair(latent_pair, is_fwd=True)

        for latent_idx, (latent_base, latent_model_input) in self.iter_latent_pair(i, latent_pair, is_fwd=True):
            # iterate over both latents in the pair. order is defined by edict.
            latent_pair[latent_idx] = self.step_forward_single(latent_idx, latent_base, latent_model_input, t, context, guidance_scale_fwd)

        # print(latent_pair[0].mean().item(), latent_pair[1].mean().item())

        return latent_pair, None

    def step_backward(self, latent: List[torch.Tensor], t: torch.Tensor, context: torch.Tensor, guidance_scale_bwd: None=None) -> Tuple[List[torch.Tensor], None]:
        latent_pair = latent
        guidance_scale_bwd = guidance_scale_bwd or self.guidance_scale_bwd
        i = self.bwd_t_to_i[t.item()]

        for latent_idx, (latent_base, latent_model_input) in self.iter_latent_pair(i, latent_pair, is_fwd=False):
            # iterate over both latents in the pair. order is defined by edict.
            latent_pair[latent_idx] = self.step_backward_single(latent_idx, latent_base, latent_model_input, t, context, guidance_scale_bwd)

        # synchronize latent pair to avoid divergance of latent pair
        latent_pair = self.sync_latent_pair(latent_pair, is_fwd=False)

        # print(latent_pair[0].mean().item(), latent_pair[1].mean().item())

        return latent_pair, None

    def get_timesteps_forward(self) -> torch.Tensor:
        # cut off steps
        return super().get_timesteps_forward()[:-self.t_limit]

    def get_timesteps_backward(self) -> torch.Tensor:
        # start later
        return super().get_timesteps_backward()[self.t_limit:]

    def encode(self, image: torch.Tensor) -> List[torch.Tensor]:
        latent = super().encode(image)
        # initialize latent pair with two copies of same latent
        latent_pair = [latent.clone(), latent.clone()]
        return latent_pair

    def decode(self, latent: List[torch.Tensor]) -> torch.Tensor:
        latent = torch.cat(latent)
        return super().decode(latent)

    def cat_latent(self, latents: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        # handle latent pair separately
        num_latent_pairs = len(latents)
        num_latents_per_pair = len(latents[0])
        assert num_latents_per_pair == 2

        return [torch.cat([latents[i][p] for i in range(num_latent_pairs)]) for p in range(num_latents_per_pair)]
