from modules.inversion.ddpm_inversion import DDPMInversion
from modules.inversion.edict_inversion import EdictInversion
from modules.inversion.null_text_inversion import NullTextInversion
from modules.inversion.proximal_negative_prompt_inversion import ProximalNegativePromptInversion
from utils.debug_utils import enable_deterministic
enable_deterministic()

import gradio as gr


class Demo:
    def __init__(self, editor_manager) -> None:
        self.models = dict([
            ("CompVis/stable-diffusion-v1-4", "SD-1.4"),
        ])

        self.inverters = dict([
            ("diffinv", "Diffusion inversion"),
            ("nti", "Null-text inversion"),
            ("npi", "Negative prompt inversion"),
            ("proxnpi", "Proximal negative prompt inversion"),
            ("edict", "EDICT"),
            ("ddpminv", "DDPM inversion"),
        ])

        self.editors = dict([
            ("simple", "Simple"),
            ("ptp", "Prompt-to-prompt"),
            ("masactrl", "MasaControl"),
            ("pnp", "Plug-and-play"),
            ("pix2pix_zero", "Pix2pix zero"),
        ])

        self.schedulers = dict([
            ("ddim", "DDIM"),
            ("dpm", "DPM"),
        ])

        self.default_values = {
            "model": "CompVis/stable-diffusion-v1-4",
            "inverter": "diffinv",
            "editor": "simple",
        }

        self.editor_manager = editor_manager
        self.demo = None
        self.build()

    def get_model_choices(self):
        return list(zip(self.models.values(), self.models.keys()))

    def get_inverter_choices(self, model):
        if model is None:
            return []
    
        if model in ("CompVis/stable-diffusion-v1-4", ):
            out = ["diffinv", "nti", "npi", "proxnpi", "edict", "ddpminv"]
        else:
            out = []

        return [(self.inverters[o], o) for o in out]

    def get_editor_choices(self, inverter):
        if inverter is None:
            return []
        
        inverter = inverter
        out = ["simple", "ptp", "masactrl", "pnp", "pix2pix_zero"]

        return [(self.editors[o], o) for o in out]
    
    def get_scheduler_choices(self):
        return [(self.schedulers[o], o) for o in self.schedulers.keys()]
        
    def run(self, cfg):
        edit_res = self.editor_manager.run(cfg)
        
        return edit_res["edit_image"]

    def update_inversion_config_visibility(self, names, cat):
        return [gr.update(visible=name.startswith(f"inverter.methods.{cat}.")) for name in names]

    def build_source_edit_image(self):
        with gr.Row():
            self.inputs["edit_cfg.source_image"] = gr.Image(label="Input", value="test/data/gnochi_mirror_sq.png", width=512, height=512)
            self.outputs["edit_image"] = gr.Image(label="Output", width=512, height=512)

    def build_model(self):
        with gr.Row():
            self.inputs["model.type"] = gr.Dropdown(label="Model", choices=self.get_model_choices(), value=self.default_values["model"])

    def build_invert(self):
        with gr.Row():
            self.inputs["inverter.type"] = gr.Dropdown(label="Inversion method", choices=self.get_inverter_choices(self.default_values["model"]), value=self.default_values["inverter"], interactive=True)

            for inverter in self.inverters:
                k = f"inverter.methods.{inverter}"
                visible = self.default_values["inverter"] == inverter

                if inverter not in ("edict", "ddpminv"):
                    self.inputs[f"{k}.scheduler"] = gr.Dropdown(
                        label="Scheduler", choices=self.get_scheduler_choices(), value="ddim", visible=visible)
                
                dft_steps = 50

                if inverter == "edict":
                    dft_fwd_cfg, dft_bwd_cfg = 3.0, 3.0
                elif inverter == "ddpminv":
                    dft_fwd_cfg, dft_bwd_cfg = 3.5, 15
                else:
                    dft_fwd_cfg, dft_bwd_cfg = 1.0, 7.5

                self.inputs[f"{k}.num_inference_steps"] = gr.Number(label="Steps", value=dft_steps, precision=0, visible=visible)
                self.inputs[f"{k}.guidance_scale_fwd"] = gr.Number(label="Forward CFG scale", value=dft_fwd_cfg, visible=visible)
                self.inputs[f"{k}.guidance_scale_bwd"] = gr.Number(label="Backward CFG scale", value=dft_bwd_cfg, visible=visible)

                if inverter == "nti":
                    self.inputs[f"{k}.num_inner_steps"] = gr.Number(
                        label="Inner steps", value=NullTextInversion.dft_num_inner_steps, precision=0, visible=visible)
                    
                    self.inputs[f"{k}.early_stop_epsilon"] = gr.Number(
                        label="Early stop eps", value=NullTextInversion.dft_early_stop_epsilon, visible=visible)
                    
                if inverter == "proxnpi":
                    self.inputs[f"{k}.prox"] = gr.Dropdown(
                        label="Prox", choices=["l0", "l1"], value="l0", visible=visible)
                    self.inputs[f"{k}.quantile"] = gr.Number(
                        label="Quantile", value=ProximalNegativePromptInversion.dft_quantile, visible=visible)
                    self.inputs[f"{k}.recon_lr"] = gr.Number(
                        label="Recon LR", value=ProximalNegativePromptInversion.dft_recon_lr, precision=0, visible=visible)
                    self.inputs[f"{k}.recon_t"] = gr.Number(
                        label="Recon t", value=ProximalNegativePromptInversion.dft_recon_t, precision=0, visible=visible)
                    self.inputs[f"{k}.dilate_mask"] = gr.Number(
                        label="Dilate Mask", value=ProximalNegativePromptInversion.dft_dilate_mask, precision=0, visible=visible)
                    
                if inverter == "edict":
                    self.inputs[f"{k}.mix_weight"] = gr.Number(
                        label="Mix weight", value=EdictInversion.dft_mix_weight, visible=visible)
                    self.inputs[f"{k}.leapfrog_steps"] = gr.Checkbox(
                        label="Leapfrog steps", value=EdictInversion.dft_leapfrog_steps, visible=visible)
                    self.inputs[f"{k}.init_image_strength"] = gr.Number(
                        label="Init image strength", value=EdictInversion.dft_init_image_strength, visible=visible)
                    
                if inverter == "ddpminv":
                    self.inputs[f"{k}.skip_steps"] = gr.Number(
                        label="Skip steps", value=DDPMInversion.dft_skip_steps, visible=visible, 
                        info="How many percent of steps to skip. Must be between 0 or 1.")
                    self.inputs[f"{k}.forward_seed"] = gr.Number(
                        label="Seed", value=-1, precision=0, visible=visible, 
                        info="Use -1 for random seed.")

    def build_edit(self):
        with gr.Column():
            self.inputs["editor.type"] = gr.Dropdown(label="Edit method", choices=self.get_editor_choices(self.default_values["inverter"]), value=self.default_values["editor"], interactive=True)
            self.inputs["edit_cfg.source_prompt"] = gr.Textbox(label="Source prompt", value="a cat sitting next to a mirror")
            self.inputs["edit_cfg.target_prompt"] = gr.Textbox(label="Target prompt", value="a tiger sitting next to a mirror")

    def build_run(self):
        self.controls["edit"] = gr.Button("Edit")

    def setup_events(self):
        self.inputs["model.type"].change(lambda model: gr.update(choices=self.get_inverter_choices(model)), inputs=self.inputs["model.type"], outputs=self.inputs["inverter.type"])
        self.inputs["inverter.type"].change(lambda inverter: gr.update(choices=self.get_editor_choices(inverter)), inputs=self.inputs["inverter.type"], outputs=self.inputs["editor.type"])

        inversion_inputs = {k: v for k, v in self.inputs.items() if k.startswith("inverter.methods.")}
        self.inputs["inverter.type"].change(lambda value: self.update_inversion_config_visibility(list(inversion_inputs.keys()), value), inputs=self.inputs["inverter.type"], outputs=list(inversion_inputs.values()))

        self.controls["edit"].click(fn=lambda *values: self.run(dict(zip(self.inputs.keys(), values))), inputs=list(self.inputs.values()), outputs=self.outputs["edit_image"])

    def build(self):
        # TODO: add gallery with old images

        # TODO: auto generate prompt and fill textbox
        with gr.Blocks() as self.demo:
            self.inputs = {}
            self.outputs = {}
            self.controls = {}

            self.build_source_edit_image()

            with gr.Row():
                self.build_edit()

                with gr.Column():
                    self.build_model()
                    self.build_invert()

            self.build_run()
            self.setup_events()

    def launch(self):
        return self.demo.launch(share=False)