
from pathlib import Path
import cv2
import yaml
import numpy as np
from itertools import product

from dataset import load_dataset

from typing import Dict, Any, Optional, Union, Tuple, List


def get_save_dir(name: str) -> Path:
    """Get default path to result save directory.

    Args:
        name (str): Name of the experiment

    Returns:
        Path: Path to the experiment
    """
    path = Path("result") / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cfg_to_filename(cfg: Dict[str, Any]) -> str:
    """Make a filename from a configuration

    Args:
        cfg (Dict[str, Any]): Configuration to convert to filename

    Returns:
        str: Filename
    """

    if isinstance(cfg, dict):
        # return "_".join([f"{k}-{_cfg_to_filename(v)}" for k, v in sorted(cfg.items())])
        return "_".join([f"{_cfg_to_filename(v)}" for k, v in sorted(cfg.items())])
    else:
        return str(cfg)


def create_configs(cfg_all: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Create sub configs from a global config file. Creates sub configs by making all
    combinations of data, editing method and method. Useful for grid search.

    Args:
        cfg_all (Union[Dict[str, Any], str]): Path to config file.

    Returns:
        Tuple[List[Dict[str, Any]], Dict[str, Any]]: Loaded global config and sub configs
    """

    if not cfg_all.endswith(".yaml"):
        # if just a name is passed instead of a path, load from default path
        cfg_all = f"cfg/eval/{cfg_all}.yaml"

    exp_name = Path(cfg_all).stem

    with open(cfg_all, "r") as f:
        cfg_all = yaml.safe_load(f)

    # keys for which to perform a carthesian product to create sub configs
    keys_batch = ["data", "edit_cfg", "method", "edit_method"]

    # product
    vals_batch = list(product(*[cfg_all.get(k, [None]) for k in keys_batch]))
    
    # create sub configs
    cfgs = [{
        **dict(zip(keys_batch, vals)),
        **{k: v for k, v in cfg_all.items() if k not in keys_batch}
    } for vals in vals_batch]

    # set paths for sub configs
    for i, cfg in enumerate(cfgs):
        sub_dir = f"{i:02d}_" + _cfg_to_filename(cfg)
        path = Path("result") / exp_name / sub_dir
        cfg["path"] = str(path)

    # set path for global config
    cfg_all["path"] = str(Path("result") / exp_name)

    return cfgs, cfg_all


class EditResultData:
    """Helper class for dataset evaluation. Takes an existing dataset and adds functions to load and store metrics.
    """

    def __init__(self, data_name: str, method: Dict[str, Any], edit_method: Dict[str, Any], 
                 edit_cfg: Dict[str, Any]=None, exp_name: Optional[str]=None, path: Optional[str]=None, 
                 skip_existing=False, **kwargs) -> None:
        """Initialies a new dataset for evaluation.

        Args:
            data_name (str): Name of the dataset to evaluate
            method (Dict[str, Any]): Inversion method configuration
            edit_method (Dict[str, Any]): Editing method configuration
            edit_cfg (Dict[str, Any], optional): Additional editing configuration (for ptp). Defaults to None.
            exp_name (Optional[str], optional): Optional experiment name, either exp_name or path should be provided. Defaults to None.
            path (Optional[str], optional): Optional path to the experiment, either exp_name or path should be provided. Defaults to None.
            skip_existing (bool): If true returns None in __getitem__ when edit image file already exists. Defaults to True.
        """

        self.path = Path(path) if path is not None else None
        self.data_name = data_name
        self.data = load_dataset(data_name, **kwargs)  # load dataset for evaluation
        self.method = method
        self.edit_method = edit_method
        self.metrics = {}
        self.exp_name = exp_name
        self.edit_cfg = edit_cfg
        self.skip_existing = skip_existing

    @staticmethod
    def from_state_dict(dic: Dict[str, Any], **kwargs) -> "EditResultData":
        """Creates a new dataset for evalation from a state dict

        Args:
            dic (Dict[str, Any]): state dict

        Returns:
            EditResultData: data
        """

        dic = {**dic}
        dic["data_name"] = dic.pop("data")
        return EditResultData(**dic, **kwargs)

    @staticmethod
    def from_metrics(eval_dir: str, categories: Optional[Dict[str, List[int]]]=None, **kwargs) -> "EditResultData":
        """Create a result dataset and load metrics

        Args:
            eval_dir (str): Path to evaluation directory.
            categories (Optional[Dict[str, List[int]]], optional): Divides images in categories and computes mean per category (e.g., for PIE categories). Defaults to None.

        Returns:
            EditResultData: Dataset with loaded metrics
        """

        # get config and metric yaml files
        cfg_file = Path(eval_dir) / "cfg.yaml"
        metric_files = (Path(eval_dir) / "metrics").glob("*.yaml")

        with open(cfg_file, "r") as f:
            cfg = yaml.safe_load(f)

        # create a experiment name
        cfg["exp_name"] = Path(eval_dir).parent.parent.stem + "_" + Path(eval_dir).parent.stem

        # load data
        data = EditResultData.from_state_dict(cfg, **kwargs)

        metrics = {}

        # load metrics
        for metric_file in metric_files:
            with open(metric_file, "r") as f:
                metric_data = yaml.safe_load(f)

            if categories is None:
                metrics_total = {"mean": metric_data["mean"]}
            else:
                values = np.array([r["value"] for r in metric_data["results"]]).astype(float)
                # recompute per category mean
                metrics_total = {"mean": {name: np.mean(values[ind]) if len(values) > 0 else None for name, ind in categories.items()}}

            metrics[metric_data["name"]] = {
                **metrics_total,
                "results": metric_data["results"],
            }

        data.metrics = metrics

        return data

    def __len__(self) -> int:
        """Dataset size

        Returns:
            int: size
        """
        return len(self.data)

    def get_edit_image_name(self, i: int) -> str:
        """Create a filename for an edited image

        Args:
            i (int): Sample index

        Returns:
            str: Edit image filename
        """

        source_prompt = self.data[i]["source_prompt"]
        target_prompt = self.data[i]["edit"]["target_prompt"]

        return f"{i:04d}_{source_prompt}_{target_prompt}"
    
    def get_metrics(self, i: int) -> Union[Dict[str, Any], None]:
        """Load per image metrics for the current sample

        Args:
            i (int): Sample index

        Returns:
            Union[Dict[str, Any], None]: Metrics dict or None in case of failure
        """

        filename = self.get_edit_image_name(i)

        metrics = {}

        if self.metrics is not None:
            for k in self.metrics.keys():
                # assert that path matches
                assert Path(filename).stem == Path(self.metrics[k]["results"][i]["file"]).stem, (Path(filename).stem + "=" + Path(self.metrics[k]["results"][i]["file"]).stem)
                
                # get metric value
                metrics[k] = self.metrics[k]["results"][i]["value"]
    
            return metrics
        else:
            return None

    def __getitem__(self, i: int) -> Dict[str, Any]:
        """Load image and metrics for an item

        Args:
            i (int): Sample index

        Returns:
            Dict[str, Any]: Image data with metrics
        """

        edit_image_file = self.path / "imgs" / f"{self.get_edit_image_name(i)}.png"

        if self.skip_existing and edit_image_file.exists():
            return None

        sample = {**self.data[i]}

        sample["image"] = sample["image"]
        sample["edit_image_file"] = edit_image_file

        if not self.data.skip_img_load and Path(sample["edit_image_file"]).exists():
            sample["edit_image"] = cv2.cvtColor(cv2.imread(str(sample["edit_image_file"])), cv2.COLOR_BGR2RGB) 
        else:
            sample["edit_image"] = None
            
        sample["metrics"] = self.get_metrics(i)

        return sample