
import torch
from collections import namedtuple
from diffusers import DPMSolverMultistepScheduler
from typing import Dict, Any

from .diffusion_inverse_scheduler import DiffusionInverseScheduler


class DPMSolverMultistepInverseScheduler(DiffusionInverseScheduler):
    """DPM inverse scheduler for DPM solver inversion.
    Based on diffusers DPMSolverMultistepInverseScheduler
    """

    Output = namedtuple("DPMSolverMultistepInverseSchedulerOutput", ("prev_sample", ))

    def __init__(self, cfg: Dict[str, Any], inv_steps: str="samesame") -> None:
        """_summary_

        Args:
            cfg (Dict[str, Any]): Original DPM scheduler configuration
            inv_steps (str, optional): Modifies at which timestep to begin for unet and scheduler.
            Valid options are "sameshift", "samesame", "shiftshift". Defaults to "samesame".
        """

        # base on diffusers implementation
        from diffusers import DPMSolverMultistepInverseScheduler as DPMSolverMultistepInverseScheduler_
        
        super().__init__()
        assert inv_steps in ("samesame", "sameshift", "shiftshift")

        self.sched = DPMSolverMultistepInverseScheduler_.from_config(cfg)
        self.inv_steps = inv_steps

    @staticmethod
    def from_scheduler(scheduler: DPMSolverMultistepScheduler, inv_steps: str="samesame", **kwargs) -> "DPMSolverMultistepInverseScheduler":
        """Creates a new DPM inverse scheduler form an existing DPM scheduler

        Args:
            scheduler (DPMSolverMultistepScheduler): Original DPM scheduler to create an inverse from
            inv_steps (str, optional): Modifies at which timestep to begin for unet and scheduler.
            Valid options are "sameshift", "samesame", "shiftshift". Defaults to "samesame".

        Returns:
            DPMSolverMultistepInverseScheduler: DPM inverse scheduler
        """
        return DPMSolverMultistepInverseScheduler(
            {**scheduler.config, **kwargs},
            inv_steps=inv_steps,
        )

    def set_timesteps(self, num_inference_steps: int) -> None:
        # get timesteps from base implementation
        self.sched.set_timesteps(num_inference_steps)

        steps = self.sched.timesteps

        assert steps[0] == 0

        if self.inv_steps == "shiftshift":
            # shift timesteps in case
            steps = torch.cat([torch.tensor(self.get_first_neg_step())[None], steps[:-1]])

        self.sched.timesteps = steps

        # self.timesteps = self.sched.timesteps
        # assert self.timesteps[0] == 0
        # assert len(self.timesteps) == num_inference_steps, "setting timesteps not supported right now"

    @property
    def timesteps(self) -> torch.Tensor:
        return self.sched.timesteps

    def get_first_neg_step(self) -> int:
        """Gets step before first step. Needed for shifting

        Returns:
            int: Step before first step
        """
        return (self.sched.timesteps[0] - (self.sched.timesteps[1] - self.sched.timesteps[0]))


    def step(self, noise_pred: torch.Tensor, t: int, latent: torch.Tensor) -> "DPMSolverMultistepInverseScheduler.Output":
        """Perform a scheduler step and update the current latent.
        Based on diffusers DPMSolverMultistepInverseScheduler.step

        Args:
            noise_pred (torch.Tensor): Noise prediction from the diffusion model
            t (int): Timestep
            latent (torch.Tensor): Current latent

        Returns:
            DPMSolverMultistepInverseScheduler.Output: Result with updated latent
        """

        model_output, sample = noise_pred, latent

        if self.sched.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.sched.timesteps.device)
        step_index = (self.sched.timesteps == timestep).nonzero()
        if len(step_index) == 0:
            step_index = len(self.sched.timesteps) - 1
        else:
            step_index = step_index.item()

        if self.inv_steps in ("sameshift",):
            # shift timesteps one back
            step_index = step_index - 1
            timestep = self.sched.timesteps[step_index] if step_index >= 0 else self.get_first_neg_step()

        prev_timestep = (
            self.sched.noisiest_timestep if step_index == len(self.sched.timesteps) - 1 else self.sched.timesteps[step_index + 1]
        )

        lower_order_final = (
            (step_index == len(self.sched.timesteps) - 1) and self.sched.config.lower_order_final and len(self.sched.timesteps) < 15
        )
        lower_order_second = (
            (step_index == len(self.sched.timesteps) - 2) and self.sched.config.lower_order_final and len(self.sched.timesteps) < 15
        )

        print(timestep.item(), "->", torch.tensor(prev_timestep).item())

        model_output = self.sched.convert_model_output(model_output, timestep, sample)
        for i in range(self.sched.config.solver_order - 1):
            self.sched.model_outputs[i] = self.sched.model_outputs[i + 1]
        self.sched.model_outputs[-1] = model_output

        if self.sched.config.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"]:
            assert False
            # noise = randn_tensor(
            #     model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
            # )
        else:
            noise = None

        if self.sched.config.solver_order == 1 or self.sched.lower_order_nums < 1 or lower_order_final:
            prev_sample = self.sched.dpm_solver_first_order_update(
                model_output, timestep, prev_timestep, sample, noise=noise
            )
        elif self.sched.config.solver_order == 2 or self.sched.lower_order_nums < 2 or lower_order_second:
            timestep_list = [self.sched.timesteps[step_index - 1], timestep]
            prev_sample = self.sched.multistep_dpm_solver_second_order_update(
                self.sched.model_outputs, timestep_list, prev_timestep, sample, noise=noise
            )
        else:
            timestep_list = [self.sched.timesteps[step_index - 2], self.sched.timesteps[step_index - 1], timestep]
            prev_sample = self.sched.multistep_dpm_solver_third_order_update(
                self.sched.model_outputs, timestep_list, prev_timestep, sample
            )

        if self.sched.lower_order_nums < self.sched.config.solver_order:
            self.sched.lower_order_nums += 1

        return DPMSolverMultistepInverseScheduler.Output(prev_sample=prev_sample)