# Metrics

## Inversion

### Mean Squared Error (MSE)

[[Wiki](https://en.wikipedia.org/wiki/Mean_squared_error)]

### Peak Signal-to-Noise Ratio (PSNR)

[[Wiki](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)]

### Learned Perceptual Image Patch Similarity (LPIPS)

[[Github](https://github.com/richzhang/PerceptualSimilarity)]

### Structural Similarity Index Measure (SSIM)

[[Wiki](https://en.wikipedia.org/wiki/Structural_similarity)]

## Editing

### CLIP similarity

[[Github](https://github.com/OpenAI/CLIP)]

#### Text<sub>tar</sub> - Image<sub>tar</sub> CLIP similarity

#### Image<sub>src</sub> - Image<sub>tar</sub> CLIP similarity

> We plot the tradeoff between two metrics, **cosine similarity of CLIP image embeddings** (how much the edited image agrees with the input image) and the directional CLIP similarity (how much the change in text captions agrees with the change in the images).
>
> -- <cite>[InstructPix2Pix](ip2p.md)</cite>

#### Text<sub>tar</sub> - caption from BLIP(Image<sub>tar</sub>) CLIP similarity

> Second, several works have analyzed the existence of a modality gap between CLIP’s image and text embeddings. To overcome this gap, we consider an additional text only metric.
> Given the generated images, we generate matching **image captions using a pre-trained BLIP** image-captioning model. We then compute the average **CLIP similarity between the prompts and captions.**
> 
> -- <cite>[Attend-and-Excite](https://arxiv.org/abs/2301.13826)</cite>

### directional CLIP similarity

> We plot the tradeoff between two metrics, cosine similarity of CLIP image embeddings (how much the edited image agrees with the input image) and the **directional CLIP similarity (how much the change in text captions agrees with the change in the images).**
>
> -- <cite>[InstructPix2Pix](ip2p.md)</cite>

### CLIP accuracy

> We measure the extent of the edit applied with CLIP Accuracy, which calculates the percentage of instances where the edited image has a higher similarity to the target text, as measured by CLIP, than to the original source text.
>
> -- <cite>[pix2pix-zero](p2p-zero.md)</cite>

### DINO

### DINOv2

### LPIPS

[[Github](https://github.com/richzhang/PerceptualSimilarity)]

### NS-LPIPS

[[Github](https://github.com/sen-mao/StyleDiffusion#ns-lpips)]

### BG-LPIPS

### MS-SSIM

[[Github](https://github.com/VainF/pytorch-msssim)]

### Fréchet Inception Distance (FID)

[[Wiki](https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance)]