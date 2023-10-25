from utils.debug_utils import enable_deterministic
enable_deterministic()

import gradio as gr


class Demo:
    def __init__(self, editor_manager) -> None:
        self.models = [
            ("CompVis/stable-diffusion-v1-4", "SD-1.4"),
        ]

        self.inverters = [
            ("diffinv", "Diffusion inversion"),
            ("nti", "Null-text inversion"),
            ("npi", "Negative prompt inversion"),
            ("proxnpi", "Proximal negative prompt inversion"),
            ("edict", "EDICT"),
            ("ddpminv", "DDPM inversion"),
        ]

        self.editors = [
            ("simple", "Simple"),
            ("ptp", "Prompt-to-prompt"),
            ("masactrl", "MasaControl"),
            ("pnp", "Plug-and-play"),
            ("pix2pix_zero", "Pix2pix zero"),
        ]

        self.default_choices = {
            "model": self.value_to_choice(self.models, "CompVis/stable-diffusion-v1-4"),
            "inverter": self.value_to_choice(self.inverters, "diffinv"),
            "editor": self.value_to_choice(self.editors, "simple"),
        }

        self.editor_manager = editor_manager
        self.demo = None
        self.build()

    def choice_to_value(self, lst, name):
        out = next((v for v, n in lst if n == name), None)
        assert out is not None, f"{name} not found in {lst}"
        return out


    def value_to_choice(self, lst, value):
        out = next((n for v, n in lst if v == value), None)
        assert out is not None, f"{value} not found {lst}"
        return out


    def get_model_choices(self):
        return list(zip(*self.models))[1]


    def get_inverter_choices(self, model):
        if model is None:
            return []

        model = self.choice_to_value(self.models, model)
    
        if model in ("CompVis/stable-diffusion-v1-4", ):
            out = ["diffinv", "nti", "npi", "proxnpi", "edict", "ddpminv"]
        else:
            out = []

        return [self.value_to_choice(self.inverters, val) for val in out]


    def get_editor_choices(self, inverter):
        if inverter is None:
            return []
        
        inverter = self.choice_to_value(self.inverters, inverter) if inverter is not None else None

        out = ["simple", "ptp", "masactrl", "pnp", "pix2pix_zero"]

        return [self.value_to_choice(self.editors, val) for val in out]
        
    def run(self, cfg):
        cfg["model.type"] = self.choice_to_value(self.models, cfg["model.type"])
        cfg["inverter.type"] = self.choice_to_value(self.inverters, cfg["inverter.type"])
        cfg["editor.type"] = self.choice_to_value(self.editors, cfg["editor.type"])

        edit_res = self.editor_manager.run(cfg)
        
        return edit_res["edit_image"]

    def update_inversion_config_visibility(self, names, cat):
        cat = self.choice_to_value(self.inverters, cat)
        return [gr.update(visible=name.startswith(f"inverter.methods.{cat}.")) for name in names]

    def build_source_edit_image(self):
        with gr.Row():
            self.inputs["edit_cfg.source_image"] = gr.Image(label="Input", value="test/data/gnochi_mirror_sq.png", width=512, height=512)
            self.outputs["edit_image"] = gr.Image(label="Output", width=512, height=512)

    def build_edit(self):
        with gr.Column():
            self.inputs["editor.type"] = gr.Dropdown(label="Edit method", choices=self.get_editor_choices(self.default_choices["inverter"]), value=self.default_choices["editor"], interactive=True)
            self.inputs["edit_cfg.source_prompt"] = gr.Textbox(label="Source prompt", value="a cat sitting next to a mirror")
            self.inputs["edit_cfg.target_prompt"] = gr.Textbox(label="Target prompt", value="a tiger sitting next to a mirror")

    def build_model(self):
        with gr.Row():
            self.inputs["model.type"] = gr.Dropdown(label="Model", choices=self.get_model_choices(), value=self.default_choices["model"])

    def build_invert(self):
        with gr.Row():
            self.inputs["inverter.type"] = gr.Dropdown(label="Inversion method", choices=self.get_inverter_choices(self.default_choices["model"]), value=self.default_choices["inverter"], interactive=True)

            for inverter, inverter_name in self.inverters:
                visible = self.default_choices["inverter"] == inverter_name
                self.inputs[f"inverter.methods.{inverter}.num_inference_steps"] = gr.Number(label="Steps", value=50, precision=0, visible=visible)
                self.inputs[f"inverter.methods.{inverter}.guidance_scale_fwd"] = gr.Number(label="Forward CFG scale", value=1.0, visible=visible)
                self.inputs[f"inverter.methods.{inverter}.guidance_scale_bwd"] = gr.Number(label="Backward CFG scale", value=7.5, visible=visible)
                self.inputs[f"inverter.methods.{inverter}.scheduler"] = gr.Dropdown(label="Scheduler", choices=["ddim", "dpm"], value="ddim", visible=visible)

                

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
        return self.demo.launch(share=True)