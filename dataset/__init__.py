
from functools import partial
from pathlib import Path

from .editing_data import EditingDataset
from .pie_bench_data import PieBenchData
from .imagen_hub import ImagenHubData
from .editing_data import EditingDataset
from .coco import CocoData
from .imagen_hub import ImagenHubData
from .base import DatasetBase, DatasetSubset


def load_dataset(name: str, **kwargs) -> DatasetBase:
    """Instantiate a dataset by its name.

    Args:
        name (str): Name of the dataset to instantiate

    Returns:
        DatasetBase: Dataset object
    """

    dataset_cls = {
        "imagenetr-fake-ti2i": partial(EditingDataset, path="data/eval/plug_and_play/imagenetr-fake-ti2i"),
        "imagenetr-ti2i": partial(EditingDataset, path="data/eval/plug_and_play/imagenetr-ti2i"),
        "ptp20": partial(EditingDataset, path="data/eval/prompt-to-prompt/prompts20.yaml"),
        "ptp_debug": partial(EditingDataset, path="data/eval/prompt-to-prompt/debug.yaml"),
        "pie": PieBenchData,
        "pie_debug": partial(PieBenchData, limit=1),
        "pie_test3": partial(PieBenchData, limit=3),
        "pie_20": partial(DatasetSubset, PieBenchData, length=20),
        "pie_sub": partial(DatasetSubset, PieBenchData, indices=[
            5, 15, 175, 18, 67, 93, 19, 29, 137, 181, 149, 528, 507, 389, 
            384, 177, 162, 136, 132, 129, 14, 21, 24, 170, 173, 241, 0, 17, 25, 31, 36, 29, 412], shuffle=False),
        "imagenhub": ImagenHubData,
        "imagenhub_debug": partial(ImagenHubData, limit=1),
        "coco": CocoData,
    }.get(name, None)
    
    if dataset_cls is None:
        dataset_cls = partial(EditingDataset, Path("data/eval") / name)

    return dataset_cls(**kwargs)