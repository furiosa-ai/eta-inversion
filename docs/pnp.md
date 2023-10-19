
# Plug-and-Play Diffusion Features for Text-Driven Image-to-Image Translation (CVPR 2023)
[[arXiv](https://arxiv.org/abs/2211.12572)] [[Github](https://github.com/MichalGeyer/plug-and-play)] [[Project](https://pnp-diffusion.github.io/)]
## Datasets <a id='datasets'></a>
[[Download](https://www.dropbox.com/sh/8giw0uhfekft47h/AAAF1frwakVsQocKczZZSX6La?dl=0)]

**ImageNet-R-TI2I**
> To test our method on a wide range of guidance images, we turn to Image-Net-R, a dataset that contains various renditions of 200 classes from Image-Net. 
**We manually select 10 classes:** "Castle", "Cat", "Goldfish", "Hummingbird", "Husky", "Jeep", "Panda", "Penguin", ""Pizza", "Violin".
To avoid low-quality images, **we manually selected 3 images per class, totaling 30 guidance images.**
> Additionally, **we automatically created 5 target prompts per image.**
All the prompts share the same template: "rendition of a class", e.g. "a painting of a jeep".
rendition is one of the existing renditions in the real ImageNet-R data-set: "an art", "a cartoon", "a graphic", "a deviantart", "a painting", "a sketch", "a graffiti", "an embroidery", "an origami", "a pattern", "a sculpture", "a tattoo", "a toy", "a video-game", "a photo", "an image".
For two (out of five) target prompts per image, we changed the correct class to another object class randomly sampled from 5 related classes (to avoid completely unreasonable translations such as penguin &rarr; jeep).
> Overall, our *ImageNet-R-TI2I* benchmark contains **150 image-text pairs: 30 guidance images with 5 target prompts each.**

**Wild-TI2I**
> **We collected a diverse dataset of 148 text-image pairs,** containing different object classes (people, animals, food, landscapes) in different renditions (realistic images, drawings, solid masks, sketches and illustrations) with different levels of semantic details. 53\% of the examples consists of real guidance images that we gathered from the Web, and the rest are generated from text.

## Metrics <a id='metrics'></a>

**Text<sub>tar</sub> - Image<sub>tar</sub> CLIP similarity** <br>
[[Github](https://github.com/OpenAI/CLIP)]

**DINO-ViT self-similarity** <br>
[[Github](https://github.com/omerbt/Splice)]

> We numerically evaluate these results using two complementary metrics: **text-image CLIP cosine similarity** to quantify how well the generated images comply with the text prompt (higher is better), and distance between **DINO-ViT self-similarity,** to quantify the extent of structure preservation (lower is better).


