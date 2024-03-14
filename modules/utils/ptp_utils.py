# Refer to https://github.com/google/prompt-to-prompt/blob/main/ptp_utils.py
# Updated for new diffusers version

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import torch
from PIL import Image
import cv2
from typing import Optional, Union, Tuple, List, Dict
from IPython.display import display
from tqdm.notebook import tqdm
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from modules.utils.ptp import AttentionReweight
from numpy import ndarray
from transformers.models.clip.tokenization_clip import CLIPTokenizer


def slerp(val, low, high):
    """ taken from https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/4
    """
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res


def slerp_tensor(val, low, high):
    shape = low.shape
    res = slerp(val, low.flatten(1), high.flatten(1))
    return res.reshape(shape)

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    image = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    image[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(image, text, (text_x, text_y ), font, 1, text_color, 2)
    return image


def view_images(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    display(pil_img)


def diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size,  model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


def text2image_ldm(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
):
    register_attention_control(model, controller)
    height = width = 256
    batch_size = len(prompt)
    
    uncond_input = model.tokenizer([""] * batch_size, padding="max_length", max_length=77, return_tensors="pt")
    uncond_embeddings = model.bert(uncond_input.input_ids.to(model.device))[0]
    
    text_input = model.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
    text_embeddings = model.bert(text_input.input_ids.to(model.device))[0]
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    context = torch.cat([uncond_embeddings, text_embeddings])
    
    model.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale)
    
    image = latent2image(model.vqvae, latents)
   
    return image, latent


def text2image_ldm_stable(
    model,
    prompt: List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    low_resource: bool = False,
):
    register_attention_control(model, controller)
    height = width = 512
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    
    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context)
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    
    # set timesteps
    extra_set_kwargs = {"offset": 1}
    model.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource)
    
    image = latent2image(model.vae, latents)
  
    return image, latent


def register_attention_control(model: StableDiffusionPipeline, controller: Optional[AttentionReweight], source_latents: None=None) -> None:
    # source_latents from 0 to T

    class CAForward:
        def __init__(self, net, place_in_unet):
            self.net = net
            self.place_in_unet = place_in_unet
            self.old_forward = net.forward  # store old function to restore later

        def __call__(self, x, encoder_hidden_states=None, attention_mask=None, context=None, mask=None):
            place_in_unet = self.place_in_unet
            self = self.net

            to_out = self.to_out
            if type(to_out) is torch.nn.modules.container.ModuleList:
                to_out = self.to_out[0]
            else:
                to_out = self.to_out

            # for different diffuser versions
            if encoder_hidden_states is not None:
                context = encoder_hidden_states
            if attention_mask is not None:
                mask = attention_mask

            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)

            if hasattr(self, "reshape_heads_to_batch_dim"):  # old diffusers
                q = self.reshape_heads_to_batch_dim(q)
                k = self.reshape_heads_to_batch_dim(k)
                v = self.reshape_heads_to_batch_dim(v)
            else:
                q = self.head_to_batch_dim(q)
                k = self.head_to_batch_dim(k)
                v = self.head_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            # print(torch.sum(attn.reshape(attn.shape[0], -1), 1))
            # attn_old = attn.clone()
            attn = controller(attn, is_cross, place_in_unet)
            # print([torch.equal(attn[i], attn_old[i]) for i in range(attn.shape[0])])
            # assert all([torch.equal(attn[i], attn_old[i]) for i in range(attn.shape[0])][:24])
            out = torch.einsum("b i j, b j d -> b i d", attn, v)

            if hasattr(self, "reshape_batch_dim_to_heads"):  # old diffusers
                out = self.reshape_batch_dim_to_heads(out)
            else:
                out = self.batch_to_head_dim(out)

            return to_out(out)
        
    # class DummyController:

    #     def __call__(self, *args):
    #         return args[0]

    #     def __init__(self):
    #         self.num_att_layers = 0

    if controller is not None:
        # override unet forward to ptp forward
        model.unet.forward = PtPUnetForward(model, source_latents)
    else:
        # reset to original unet forward if controller is None
        model.unet.forward = model.unet.forward.unet_forward

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ in ('Attention', 'CrossAttention'):
            if controller is not None:
                net_.forward = CAForward(net_, place_in_unet)
            else:
                net_.forward = net_.forward.old_forward
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    assert cross_att_count == 32, "verify correct diffusers version and that model is SD"

    if controller is not None:
        controller.num_att_layers = cross_att_count

    
def get_word_inds(text: str, word_place: int, tokenizer: CLIPTokenizer) -> ndarray:
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def update_alpha_time_word(alpha: torch.Tensor, bounds: Union[float, Tuple[float, float]], prompt_ind: int,
                           word_inds: Optional[torch.Tensor]=None) -> torch.Tensor:
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(prompts: List[str], num_steps: int,
                                   cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
                                   tokenizer: CLIPTokenizer, max_num_words: int=77) -> torch.Tensor:
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],
                                                  i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
             inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
             for i, ind in enumerate(inds):
                 if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words


class PtPUnetForward:
    """Place-in UNet forward function using Ptp
    """

    def __init__(self, model: StableDiffusionPipeline, source_latents: List[torch.Tensor]) -> None:
        """Initializes a new Ptp UNet forward function

        Args:
            model (StableDiffusionPipeline): SD model
            source_latents (List[torch.Tensor]): Optionally use source latents from inverse process instead of the source latents from denoising
        """

        self.model = model
        self.unet_forward = model.unet.forward
        self.source_latents = source_latents
        self.idx = 0

    def __call__(self, sample: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor) -> UNet2DConditionOutput:
        if self.source_latents is not None:
            assert sample.shape[0] == 4
            # use stored latents from forward and patch in
            sample[0::2] = self.source_latents[-(self.idx + 1)]
            self.idx += 1

        return self.unet_forward(sample, timestep, encoder_hidden_states=encoder_hidden_states)
