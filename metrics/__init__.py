
# from .metrics import MSEMetric
# from .metrics import PSNRMetric
from .metrics import LPIPSMetric
from .clip_similarity import CLIPSimilarity
from .dino_vit_structure import DinoVitStructure


# def load_metric(name, *args, **kwargs):
#     return {
#         "mse": MSEMetric,
#         "psnr": PSNRMetric,
#         "lpips": LPIPSMetric,
#     }[name](*args, **kwargs)
