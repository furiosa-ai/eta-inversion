
from functools import partial
from .editing_data import EditingDataset
from .pie_bench_data import PieBenchData
from .imagen_hub import ImagenHubData
from .editing_data import EditingDataset
from .imagen_hub import ImagenHubData
from .base import DatasetBase


def load_dataset(name: str, **kwargs) -> DatasetBase:
    """Instantiate a dataset by its name.

    Args:
        name (str): Name of the dataset to instantiate

    Returns:
        DatasetBase: Dataset object
    """
    return {
        "imagenetr-fake-ti2i": partial(EditingDataset, path="data/eval/plug_and_play/imagenetr-fake-ti2i"),
        "imagenetr-ti2i": partial(EditingDataset, path="data/eval/plug_and_play/imagenetr-ti2i"),
        "ptp20": partial(EditingDataset, path="data/eval/prompt-to-prompt/prompts20.yaml"),
        "ptp_debug": partial(EditingDataset, path="data/eval/prompt-to-prompt/debug.yaml"),
        "pie": PieBenchData,
        "pie_debug": partial(PieBenchData, limit=1),
        "pie_test3": partial(PieBenchData, limit=3),
        "imagenhub": ImagenHubData,
        "imagenhub_debug": partial(ImagenHubData, limit=1),
    }[name](**kwargs)