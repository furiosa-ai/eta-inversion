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
        
    def process_config(self, cfg):
        return cfg

    def run(self, cfg):
        cfg = self.process_config(cfg)

        edit_res = self.editor_manager.run(cfg)
        
        return edit_res["edit_image"]

    def update_config_visibility(self, type, names, name_visible):
        visibilities = [name.startswith(f"{type}.methods.{name_visible}.") for name in names]
        return [gr.update(visible=v) for v in visibilities]

    def build_source_edit_image(self):
        with gr.Row():
            self.inputs["editor.source_image"] = gr.Image(label="Input", value="test/data/gnochi_mirror_sq.png", width=512, height=512)
            self.outputs["edit_image"] = gr.Image(label="Output", width=512, height=512)

    def build_model(self):
        with gr.Row():
            self.inputs["model.type"] = gr.Dropdown(label="Model", choices=self.get_model_choices(), value=self.default_values["model"])

    def build_invert(self):
        self.groups["inverter"] = {}

        with gr.Column(), gr.Group():
            with gr.Row():
                self.inputs["inverter.type"] = gr.Dropdown(label="Inversion method", choices=self.get_inverter_choices(self.default_values["model"]), value=self.default_values["inverter"], interactive=True)
            
            for inverter in self.inverters:
                visible = self.default_values["inverter"] == inverter
                with gr.Group(visible=visible) as self.groups["inverter"][inverter]:
                    k = f"inverter.methods.{inverter}"
                    dft_steps = 50

                    if inverter == "edict":
                        dft_fwd_cfg, dft_bwd_cfg = 3.0, 3.0
                    elif inverter == "ddpminv":
                        dft_fwd_cfg, dft_bwd_cfg = 3.5, 15
                    else:
                        dft_fwd_cfg, dft_bwd_cfg = 1.0, 7.5

                    with gr.Row():
                        if inverter not in ("edict", "ddpminv"):
                            self.inputs[f"{k}.scheduler"] = gr.Dropdown(
                                label="Scheduler", choices=self.get_scheduler_choices(), value="ddim")
                        
                        self.inputs[f"{k}.num_inference_steps"] = gr.Number(label="Steps", value=dft_steps, precision=0)

                    with gr.Row():
                        self.inputs[f"{k}.guidance_scale_fwd"] = gr.Number(label="Forward CFG scale", value=dft_fwd_cfg)
                        self.inputs[f"{k}.guidance_scale_bwd"] = gr.Number(label="Backward CFG scale", value=dft_bwd_cfg)

                    with gr.Row():
                        if inverter == "nti":
                            self.inputs[f"{k}.num_inner_steps"] = gr.Number(
                                label="Inner steps", value=NullTextInversion.dft_num_inner_steps, precision=0)
                            
                            self.inputs[f"{k}.early_stop_epsilon"] = gr.Number(
                                label="Early stop eps", value=NullTextInversion.dft_early_stop_epsilon)
                            
                        if inverter == "proxnpi":
                            self.inputs[f"{k}.prox"] = gr.Dropdown(
                                label="Prox", choices=["l0", "l1"], value="l0")
                            self.inputs[f"{k}.quantile"] = gr.Number(
                                label="Quantile", value=ProximalNegativePromptInversion.dft_quantile,
                                minimum=0, maximum=1, step=0.1,)
                            self.inputs[f"{k}.recon_lr"] = gr.Number(
                                label="Recon LR", value=ProximalNegativePromptInversion.dft_recon_lr, precision=0)
                            self.inputs[f"{k}.recon_t"] = gr.Number(
                                label="Recon t", value=ProximalNegativePromptInversion.dft_recon_t, precision=0)
                            self.inputs[f"{k}.dilate_mask"] = gr.Number(
                                label="Dilate Mask", value=ProximalNegativePromptInversion.dft_dilate_mask, precision=0)
                            
                        if inverter == "edict":
                            self.inputs[f"{k}.mix_weight"] = gr.Number(
                                label="Mix weight", value=EdictInversion.dft_mix_weight, 
                                minimum=0, maximum=1, step=0.1,)
                            self.inputs[f"{k}.leapfrog_steps"] = gr.Checkbox(
                                label="Leapfrog steps", value=EdictInversion.dft_leapfrog_steps)
                            self.inputs[f"{k}.init_image_strength"] = gr.Number(
                                label="Init image strength", value=EdictInversion.dft_init_image_strength,
                                minimum=0, maximum=1, step=0.1,)
                            
                        if inverter == "ddpminv":
                            self.inputs[f"{k}.skip_steps"] = gr.Number(
                                label="Skip steps", value=DDPMInversion.dft_skip_steps, 
                                minimum=0, maximum=1, step=0.1,
                                info="How many percent of steps to skip. Must be between 0 or 1.")
                            self.inputs[f"{k}.forward_seed"] = gr.Number(
                                label="Seed", value=-1, precision=0, 
                                info="Use -1 for random seed.")

    def build_edit(self):
        self.groups["editor"] = {}

        with gr.Column(), gr.Group():
            self.inputs["editor.type"] = gr.Dropdown(label="Edit method", choices=self.get_editor_choices(self.default_values["inverter"]), value=self.default_values["editor"], interactive=True)
            
            with gr.Row():
                self.inputs["editor.source_prompt"] = gr.Textbox(label="Source prompt", value="a cat sitting next to a mirror")
                self.inputs["editor.target_prompt"] = gr.Textbox(label="Target prompt", value="a tiger sitting next to a mirror")
            # change to editor.methods.{name}.edit_cfg and move inside loop

            for editor in self.editors:
                visible = self.default_values["editor"] == editor
                with gr.Group(visible=visible) as self.groups["editor"][editor]:
                    k = f"editor.methods.{editor}"

                    if editor == "simple":
                        pass

                    if editor == "ptp":
                        # is_replace_controller
                        # cross_replace_steps = {'default_': .8,}
                        # self_replace_steps = .5
                        # blend_word = ((('cat',), ("tiger",))) # for local edit. If it is not local yet - use only the source object: blend_word = ((('cat',), ("cat",))).
                        # eq_params = {"words": ("tiger",), "values": (2,)} # amplify attention to the word "tiger" by *2 

                        with gr.Row():
                            self.inputs[f"{k}.dft_cfg.is_replace_controller"] = gr.Checkbox(
                                label="Replace", value=True
                            )

                            self.inputs[f"{k}.dft_cfg.cross_replace_steps"] = gr.Number(
                                label="Cross replace steps", value=0.8, minimum=0, maximum=1, step=0.1,
                            )

                            self.inputs[f"{k}.dft_cfg.self_replace_steps"] = gr.Number(
                                label="Self replace steps", value=0.5, minimum=0, maximum=1, step=0.1,
                            )

                        with gr.Row():
                            self.inputs[f"{k}.dft_cfg.source_blend_word"] = gr.Textbox(
                                label="Source blend word", value="cat",
                            )

                            self.inputs[f"{k}.dft_cfg.target_blend_word"] = gr.Textbox(
                                label="Target blend word", value="tiger",
                            )

                        with gr.Row():
                            self.inputs[f"{k}.dft_cfg.eq_params_words"] = gr.Textbox(
                                label="Amplify word", value="tiger",
                            )

                            self.inputs[f"{k}.dft_cfg.eq_params_values"] = gr.Number(
                                label="Amplify amount", value=2,
                            )
                    
                    if editor == "masactrl":
                        self.inputs[f"{k}.step"] = gr.Number(
                            label="Step", value=4, precision=0)
                        
                        self.inputs[f"{k}.layer"] = gr.Number(
                            label="Layer", value=10, precision=0)

                    # if editor == "pnp":
                    if editor == "pnp":
                        self.inputs[f"{k}.no_null_source_prompt"] = gr.Checkbox(
                            label="Use source prompt", value=False, info="By default PNP does not use source prompt for inversion.")

                    if editor == "pix2pix_zero":
                        self.inputs[f"{k}.cross_attention_guidance_amount"] = gr.Number(
                            label="Attention Guidance", value=0.1, minimum=0, maximum=1, step=0.1,)

    def build_run(self):
        self.controls["edit"] = gr.Button("Edit")

    def setup_events(self):
        self.inputs["model.type"].change(lambda model: gr.update(choices=self.get_inverter_choices(model)), inputs=self.inputs["model.type"], outputs=self.inputs["inverter.type"])
        self.inputs["inverter.type"].change(lambda inverter: gr.update(choices=self.get_editor_choices(inverter)), inputs=self.inputs["inverter.type"], outputs=self.inputs["editor.type"])
        self.controls["edit"].click(fn=lambda *values: self.run(dict(zip(self.inputs.keys(), values))), inputs=list(self.inputs.values()), outputs=self.outputs["edit_image"])

        self.setup_inverter_edit_config_events("inverter")
        self.setup_inverter_edit_config_events("editor")

    def setup_inverter_edit_config_events(self, category):
        dropdown = self.inputs[f"{category}.type"]
        group_names, groups = zip(*self.groups[category].items())

        def _update_vis(value):
            visibilities = [(group_name == value) for group_name in group_names]
            return [gr.update(visible=v) for v in visibilities]

        dropdown.change(
            _update_vis,
            inputs=dropdown,
            outputs=list(groups)
        )

    def build(self):
        # TODO: add gallery with old images

        # TODO: auto generate prompt and fill textbox
        with gr.Blocks() as self.demo:
            self.inputs = {}
            self.outputs = {}
            self.controls = {}
            self.groups = {}

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