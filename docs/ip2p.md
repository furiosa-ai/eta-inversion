# InstructPix2Pix: Learning to Follow Image Editing Instructions (CVPR 2023)
[[arXiv](https://arxiv.org/abs/2211.09800)] [[Github](https://www.timothybrooks.com/instruct-pix2pix)] [[Project](https://github.com/timothybrooks/instruct-pix2pix)]
## Datasets <a id='datasets'></a>
[[Download](https://instruct-pix2pix.eecs.berkeley.edu/human-written-prompts.jsonl)]

**human-written-prompts**
> To produce the fine-tuning dataset, we sampled 700 input captions from the LAION-Aesthetics V2 6.5+ dataset and manually wrote instructions and output captions.

## Methods <a id='methods'></a>



## Metrics <a id='metrics'></a>

**Image<sub>src</sub> - Image<sub>tar</sub> CLIP similarity** <br>
[[Github](https://github.com/OpenAI/CLIP)]

**directional CLIP similarity** <br>
[[Github](https://github.com/OpenAI/CLIP)]

> We plot the tradeoff between two metrics, cosine similarity of CLIP image embeddings (how much the edited image agrees with the input image) and the directional CLIP similarity (how much the change in text captions agrees with the change in the images). These are competing metrics—increasing the degree to which the output images correspond to a desired edit will reduce their similarity (consistency) with the input image.
