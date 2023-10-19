# Zero-shot Image-to-Image Translation (SIGGRAPH 2023)
[[arXiv](https://arxiv.org/abs/2302.03027)] [[Github](https://github.com/pix2pixzero/pix2pix-zero)] [[Project](https://pix2pixzero.github.io)]

## Datasets <a id='datasets'></a>

## Methods <a id='methods'></a>


![p2p-zero_methods](images/p2p-zero_methods.jpg)

## Metrics <a id='metrics'></a>

>For quantitative evaluations, we measure three criteria: (1) whether the edit was applied successfully, (2) whether the structure of the input image is retained in the edited image, and (3) if the background regions of the image stay unchanged. We measure the extent of the edit applied with CLIP Accuracy, which calculates the percentage of instances where the edited image has a higher similarity to the target text, as measured by CLIP, than to the original source text. Subsequently, the structural consistency of the edited image is measured using Structure Dist. A lower score on Structure Dist means that the structure of the edited image is more similar to the input image. Lastly, to ensure that we retain the background after edits, we calculate the background LPIPS error (BG LPIPS). This is done by measuring the LPIPS distance between the background regions of the original and edited images. The background regions are identiﬁed using the object detector Detic. A lower BG LPIPS score indicates that the background of the original image has been well preserved. The background error metric BG LPIPS is only applicable for speciﬁc editing tasks where only the foreground object needs to be altered (e.g. changing a cat to a dog, or a horse to a zebra). However, for editing tasks that involve changing the entire image (e.g. converting a sketch to an oil pastel), this metric is not relevant.