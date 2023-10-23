import contextlib
import torch
from tqdm import tqdm

from diffusers import DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler
from modules.inverse_schedulers import DiffusionInverseScheduler, DDIMInverseScheduler, DPMSolverMultistepInverseScheduler, DDPMInverseScheduler
from ..editing.controller import ControllerBase, ControllerEmpty
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union


class DiffusionInversion:
    """Main class for diffusion inversion. Supports DDIM and DPM scheduler for now.
    """

    def __init__(self, model: StableDiffusionPipeline, scheduler: Optional[str]=None, num_inference_steps: Optional[int]=None, 
                 guidance_scale_bwd: Optional[float]=None, guidance_scale_fwd: Optional[float]=None,
                 verbose: bool=False) -> None:
        """Creates a new diffusion inversion instance

        Args:
            model (StableDiffusionPipeline): The diffusion model to invert. Must be Stable Diffusion for now.
            scheduler (Optional[str], optional): Name of the scheduler to invert. 
            Possbile choices are "ddim", "dpm" and "ddpm". Defaults to "ddim".
            num_inference_steps (Optional[int], optional): Number of denoising steps. Usually set to 50. Defaults to None.
            guidance_scale_bwd (Optional[float], optional): Classifier-free guidance scale for backward process (denoising). Defaults to None.
            guidance_scale_fwd (Optional[float], optional): Classifier-free guidance scale for forward process (inversion). Defaults to None.
            verbose (bool, optional): If True, print debug messages. Defaults to False.
        """

        # initialize with default parameters of ddim inversion if unset
        scheduler = scheduler or "ddim"
        self.num_inference_steps = num_inference_steps or 50
        self.guidance_scale_bwd = guidance_scale_bwd or 7.5
        self.guidance_scale_fwd = guidance_scale_fwd or 1

        self.model = model
        self.unet = model.unet
        self.device = self.model.device
        self.verbose = verbose
        self.controller = None

        # create scheduler for model, backward process and forward process
        model.scheduler, self.scheduler_bwd, self.scheduler_fwd = self.create_schedulers(model, scheduler, self.num_inference_steps)

        # mapping of timesteps to step index
        self.bwd_t_to_i = {t.item(): i for i, t in enumerate(self.scheduler_bwd.timesteps)}
        self.fwd_t_to_i = {t.item(): i for i, t in enumerate(self.scheduler_fwd.timesteps)}

        # initialize controller to empty controller
        # used by e.g., ptp to modify latent per diffusion step
        with self.use_controller(None):
            pass

    @contextlib.contextmanager
    def use_controller(self, controller: Optional[ControllerBase]) -> Iterator[None]:
        """Apply a controller to modify the diffusion process (works backward and forward)

        Args:
            controller (Optional[ControllerBase]): Controller to use. Pass None to disable.

        Yields:
            Iterator[None]: context
        """

        if controller is None:
            controller = ControllerEmpty()

        # set controller
        self.controller = controller

        # call controller callbacks
        self.controller.begin()
        yield
        self.controller.end()

        # reset controller
        self.controller = ControllerEmpty()

    def pbar(self, it: Iterable, **kwargs) -> Iterable:
        """If verbose is enabled, display progress bar in console

        Args:
            it (Iterable): Iterator

        Returns:
            Iterable: Iterator with progress bar (if enabled)
        """
        return tqdm(it, **kwargs) if self.verbose else it

    # @contextlib.contextmanager
    # def use_config(self, **kwargs):
    #     old_cfg = {k: getattr(self, k) for k in kwargs.keys()}
    #     for k, v in kwargs.items():
    #         setattr(self, k, v)
    #     yield
    #     for k, v in old_cfg.items():
    #         setattr(self, k, v)

    def create_schedulers(self, model: StableDiffusionPipeline, scheduler: Union[str, Dict[str, Any]], num_inference_steps: int
                          ) -> Tuple[DiffusionInverseScheduler, DiffusionInverseScheduler, DiffusionInverseScheduler]:
        """Create model schedulers for model, backward process and forward process

        Args:
            model (StableDiffusionPipeline): Diffusion model
            scheduler (Union[str, Dict[str, Any]]): Scheduler name or configuration to create
            num_inference_steps (int): Number of diffusion steps

        Returns:
            Tuple[DiffusionInverseScheduler, DiffusionInverseScheduler, DiffusionInverseScheduler]: schedulers for model, backward process and forward process
        """

        # grab scheduler arguments
        if isinstance(scheduler, str):
            scheduler_name = scheduler
            scheduler_kwargs = {}
            scheduler_inv_kwargs = {}
        elif isinstance(scheduler, dict):
            scheduler_kwargs = {**scheduler}
            scheduler_name = scheduler_kwargs.pop("type")

            scheduler_inv_kwargs = {} 
            if "inv_steps" in scheduler_kwargs:
                scheduler_inv_kwargs["inv_steps"] = scheduler_kwargs.pop("inv_steps")
        else:
            raise Exception(type(scheduler))

        if scheduler_name == "ddim":
            # to match npi, nti, ... implementation, set certain default arguments
            scheduler_kwargs = {
                "clip_sample": False,
                "set_alpha_to_one": False,
                **scheduler_kwargs,
            }

        # supported schedulers
        scheduler_cls = {
            "ddim": DDIMScheduler,  # ddim solver
            "ddpm": DDIMScheduler,  # simulate ddpm with ddim and eta=1
            "dpm": DPMSolverMultistepScheduler,  # dpm solver
        }[scheduler_name]

        # create a new (backward) scheduler based on model scheduler config and provided config
        scheduler = scheduler_cls.from_config({**model.scheduler.config, **scheduler_kwargs})

        assert isinstance(scheduler, (DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler))

        scheduler_bwd = scheduler

        # set inference steps
        scheduler.set_timesteps(num_inference_steps)
        scheduler_bwd.set_timesteps(num_inference_steps)

        # create forward/inverse scheduler
        if isinstance(scheduler, DDIMScheduler):
            if scheduler_name == "ddpm":
                # in case of DDPM inversion we need a special inverse scheduler
                scheduler_fwd = DDPMInverseScheduler.from_scheduler(scheduler, **scheduler_inv_kwargs)
            else:
                # otherwise use ddim inverse scheduler implementation
                scheduler_fwd = DDIMInverseScheduler.from_scheduler(scheduler, **scheduler_inv_kwargs)
        elif isinstance(scheduler, DPMSolverMultistepScheduler):
            scheduler_fwd = DPMSolverMultistepInverseScheduler.from_scheduler(scheduler, **scheduler_inv_kwargs)

        # set inference steps
        scheduler_fwd.set_timesteps(num_inference_steps)

        assert scheduler_fwd.timesteps[0] < scheduler_fwd.timesteps[1], "wrong timestamp order, not increasing. update diffusers?"

        return scheduler, scheduler_bwd, scheduler_fwd

    @staticmethod
    def get_available_schedulers() -> List[str]:
        """Get a list of all supported scheduler names

        Returns:
            List[str]: Supported scheduler names
        """
        return ["ddim", "ddpm", "dpm"]

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """VAE-decode latent z0 to image

        Args:
            latent (torch.Tensor): z0 (output of backward diffusion process)

        Returns:
            torch.Tensor: Output image tensor
        """
        latent = 1 / 0.18215 * latent
        image = self.model.vae.decode(latent)['sample']
        return image

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """VAE-encode image to latent z0

        Args:
            image (torch.Tensor): image tensor for encoding

        Returns:
            torch.Tensor: latent z0 (input for forward diffusion process)
        """
        latent = self.model.vae.encode(image)['latent_dist'].mean
        latent = latent * 0.18215
        return latent

    def create_context(self, prompt: str, negative_prompt: str="") -> torch.Tensor:
        """Tokenize and embed prompt to use as context for Stable Diffusion. 
        Creates unconditioal and conditional context for classifier-free guidance.

        Args:
            prompt (str): Prompt to embed
            negative_prompt (str, optional): For negative prompting. Defaults to "".

        Returns:
            torch.Tensor: Context for SD's UNet. Unconditional and conditional embedding concated at the batch dim.
        """

        # tokenize and embed prompt
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]

        if negative_prompt is not None:
            # tokenize and embed negative prompt
            uncond_input = self.model.tokenizer(
                [negative_prompt], padding="max_length", max_length=self.model.tokenizer.model_max_length, return_tensors="pt"
            )
            uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]

            context = torch.cat([uncond_embeddings, text_embeddings])
        else:
            context = text_embeddings

        return context
    
    def predict_noise(self, latent: torch.Tensor, t: torch.Tensor, context: torch.Tensor, guidance_scale: Optional[Union[float, int]], is_fwd: bool=False,  **kwargs) -> torch.Tensor:
        """Make a noise prediction at timestep t using the diffusion model with classifier-free guidance.

        Args:
            latent (torch.Tensor): Current latent
            t (torch.Tensor): Timestep
            context (torch.Tensor): Prompt embeddings
            guidance_scale (Optional[Union[float, int]]): Guidance scale for classifier-free guidance. Set to None to disable.
            is_fwd (bool, optional): True, if in forward diffusion process. Defaults to False.

        Returns:
            torch.Tensor: Noise prediction
        """

        if guidance_scale is None:
            noise_pred = self.unet(latent, t, encoder_hidden_states=context, **kwargs)["sample"] 
        else:
            # perform cfg

            # duplicate latent at the batch dimension to match uncond and cond embedding in context for cfg
            if latent.shape[0] * 2 == context.shape[0]:
                latent = torch.cat([latent] * 2)
            else:
                assert latent.shape[0] == context.shape[0]

            noise_pred = self.unet(latent, t, encoder_hidden_states=context, **kwargs)["sample"]

            # cfg
            noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        
        return noise_pred

    def step_forward(self, noise_pred: torch.Tensor, t: torch.Tensor, latent: torch.Tensor, *args, **kwargs) -> Any:
        """Perform a forward step using the forward noise scheduler and predictied noise.

        Args:
            noise_pred (torch.Tensor): Predicted noise
            t (torch.Tensor): Timestep
            latent (torch.Tensor): Current latent

        Returns:
            Any: Scheduler output containing new latent.
        """
        return self.scheduler_fwd.step(noise_pred, t, latent, *args, **kwargs)

    def step_backward(self, noise_pred: torch.Tensor, t: torch.Tensor, latent: torch.Tensor, *args, **kwargs) -> Any:
        """Perform a backward step using the backward noise scheduler and predictied noise.

        Args:
            noise_pred (torch.Tensor): Predicted noise
            t (torch.Tensor): Timestep
            latent (torch.Tensor): Current latent

        Returns:
            Any: Scheduler output containing new latent.
        """
        return self.scheduler_bwd.step(noise_pred, t, latent, *args, **kwargs) 

    def predict_step_forward(self, latent: torch.Tensor, t: torch.Tensor, context: torch.Tensor, guidance_scale_fwd: Optional[float]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one forward diffusion steps. Makes a noise prediction using SD's UNet first and then updates the latent using the noise scheduler.

        Args:
            latent (torch.Tensor): Current latent
            t (torch.Tensor): Timestep
            context (torch.Tensor): Prompt embeddings
            guidance_scale_fwd (Optional[float], optional): Guidance scale for classifier-free guidance. Set to None for default default scale. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: updated latent and noise prediction
        """

        guidance_scale_fwd = guidance_scale_fwd or self.guidance_scale_fwd

        # call controller callback (e.g. ptp)
        latent = self.controller.begin_step(latent=latent)

        # make a noise prediction using UNet
        noise_pred = self.predict_noise(latent, t, context, guidance_scale_fwd, is_fwd=True)

        # update the latent based on the predicted noise with the noise schedulers
        new_latent = self.step_forward(noise_pred, t, latent).prev_sample

        # call controller callback to modify latent (e.g. ptp)
        new_latent = self.controller.end_step(latent=new_latent, noise_pred=noise_pred, t=t)

        return new_latent, noise_pred

    def predict_step_backward(self, latent: torch.Tensor, t: torch.Tensor, context: torch.Tensor, guidance_scale_bwd: Optional[float]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one backward diffusion steps. Makes a noise prediction using SD's UNet first and then updates the latent using the noise scheduler.

        Args:
            latent (torch.Tensor): Current latent
            t (torch.Tensor): Timestep
            context (torch.Tensor): Prompt embeddings
            guidance_scale_bwd (Optional[float], optional): Guidance scale for classifier-free guidance. Set to None for default default scale. Defaults to None.

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

        # call controller callback to modify latent (e.g. ptp)
        new_latent = self.controller.end_step(latent=new_latent, noise_pred=noise_pred, t=t)

        return new_latent, noise_pred

    def get_timesteps_forward(self) -> torch.Tensor:
        """Get timesteps of forward scheduler

        Returns:
            torch.Tensor: Forward timesteps
        """
        return self.scheduler_fwd.timesteps

    def get_timesteps_backward(self) -> torch.Tensor:
        """Get timesteps of backward scheduler

        Returns:
            torch.Tensor: Backward timesteps
        """
        return self.scheduler_bwd.timesteps

    def diffusion_forward(self, latent: torch.Tensor, context: torch.Tensor, guidance_scale_fwd: Optional[float]=None
                          ) -> Dict[str, Any]:
        """Run forward (inverse) diffusion process to get immediate latents, noise predictions and inverse latent zT frim latent z0

        Args:
            latent (torch.Tensor): Current latent
            context (torch.Tensor): Prompt embeddings
            guidance_scale_fwd (Optional[float], optional): Guidance scale for classifier-free guidance. Set to None for default default scale. Defaults to None.

        Returns:
            Dict[str, Any]: immediate latents, noise predictions and inverse latent zT
        """

        guidance_scale_fwd = guidance_scale_fwd or self.guidance_scale_fwd

        # collect all intermediate latens (including start latent) and noise predictions
        latents = [latent]
        noise_preds = []

        # dont compute gradient
        if isinstance(latent, torch.Tensor):
            latent = latent.clone().detach()

        for i, t in enumerate(self.pbar(self.get_timesteps_forward(), desc="forward")):
            # iterate over all timesteps and gradually invert latent
            latent, noise_pred = self.predict_step_forward(latent, t, context, guidance_scale_fwd)

            noise_preds.append(noise_pred)
            latents.append(latent)

        return {"latents": latents, "noise_preds": noise_preds, "zT_inv": latents[-1]}

    def diffusion_backward(self, latent: torch.Tensor, context: torch.Tensor, inv_result: Dict[str, Any]) -> torch.Tensor:
        """Run backward (denoise) diffusion process to get latent z0 from inverse latent zT

        Args:
            latent (torch.Tensor): latent zT obtained from inversion
            context (torch.Tensor): embedded prompt
            inv_result (Dict[str, Any]): additional results from inversion

        Returns:
            torch.Tensor: latent z0
        """

        for i, t in enumerate(self.pbar(self.get_timesteps_backward(), desc="backward")):
            # iterate over all timesteps and gradually denoise latent
            latent, noise_pred = self.predict_step_backward(latent, t, context)
            
        return latent

    def invert(self, image: torch.Tensor, prompt: Optional[str]=None, context: Optional[torch.Tensor]=None, 
               guidance_scale_fwd: Optional[float]=None) -> Dict[str, Any]:
        """Invert image to inverse latent zT.

        Args:
            image (torch.Tensor): Image tensor for inversion
            prompt (Optional[str], optional): Prompt for inversion. Must be None if context is passed. Defaults to None.
            context (Optional[torch.Tensor], optional): Context for inversion. Must be None if prompt is passed. Defaults to None.
            guidance_scale_fwd (Optional[float], optional): Classifier-free guidance scale. Defaults to None.

        Returns:
            Dict[str, Any]: Inversion result (immediate latents, noise predictions, inverse latent zT, context)
        """

        context = context if context is not None else self.create_context(prompt)

        latent = self.encode(image)
        fwd_result = self.diffusion_forward(latent, context, guidance_scale_fwd=guidance_scale_fwd)
        fwd_result["context"] = context  # add embedded prompts
        return fwd_result

    def cat_context(self, contexts: List[torch.Tensor]) -> torch.Tensor:
        """Concatentate multiple contexts to one batch while keeping unconditional and conditional embeddings separated.
        E.g., [[uncond1, cond1], [uncond2, cond2]] -> [uncond1, uncond2, cond1, cond2]

        Args:
            contexts (List[torch.Tensor]): List of contexts. Should have batch size 2 for cfg.

        Returns:
            torch.Tensor: Batched context
        """

        # dim for cfg should be first dim
        n = len(contexts)  # number of prompts
        b = contexts[0].shape[0]  # should be 2 for cfg
        assert b == 2, "Cfg should have batch dimension 2"
        x = torch.stack(contexts, 1)  # keep uncond and cond separated as b n ...
        x = x.reshape(b * n, *x.shape[2:])  # flatten first two dimensions b and n
        return x

    def cat_latent(self, latents: List[torch.Tensor]) -> torch.Tensor:
        """Concatenate multiple latents to one latent for batched diffusion with multiple prompts.

        Args:
            latents (List[torch.Tensor]): List of latents to concatenate

        Returns:
            torch.Tensor: Batched latents
        """
        # for multiple prompts
        return torch.cat(latents)

    def sample(self, inv_result: Dict[str, Any], prompt: Optional[Union[str, List[str]]]=None, 
               context: Optional[Union[torch.Tensor, List[torch.Tensor]]]=None) -> Dict[str, Any]:
        """Sample an image from the inversion result.

        Args:
            inv_result (Dict[str, Any]): Result from invert()
            prompt (Optional[Union[str, List[str]]], optional): Prompt or list of prompts for denoising. Must be None if context is passed. Defaults to None.
            context (Optional[Union[torch.Tensor, List[torch.Tensor]]], optional): Context or list of contexts for denoising. Must be None if prompt is passed. Defaults to None.

        Returns:
            Dict[str, Any]: Image and latent z0
        """
        
        latent = inv_result["latents"][-1]

        # create context from prompt(s) if not provided
        context = context if context is not None else self.create_context(prompt)

        if isinstance(context, list):
            # batch
            num_prompts = len(context)
            context = self.cat_context(context)
            latent = self.cat_latent([latent] * num_prompts)

        # denoise
        z0 = self.diffusion_backward(latent, context, inv_result)

        if z0 is None:
            return None

        # vae decode
        image = self.decode(z0)
        return {"image": image, "latent": z0}

    def invert_sample(self, image: torch.Tensor, prompt: str) -> Dict[str, Any]:
        """Invert an image and sample a new image. Combination of invert() and sample().

        Args:
            image (torch.Tensor): Image for inversion
            prompt (str): Prompt for inversion and denoising

        Returns:
            Dict[str, Any]: Inverse image result
        """
        context = self.create_context(prompt)
        inv_res = self.invert(image, context=context)
        return self.sample(inv_res, context=context)
