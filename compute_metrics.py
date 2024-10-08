
from utils.debug_utils import enable_deterministic
from utils.device_pool import DevicePool
enable_deterministic()

from pathlib import Path
import argparse
from tqdm import tqdm
import yaml

import torch
from metrics.edit_metric import EditMetric
from utils.eval_utils import EditResultData
from modules import StablePreprocess

from typing import List, Optional, Dict, Any
import gc


def run_compute_metric(metric: EditMetric, eval_dir: Path, model: str, path: str, data: str, method: Dict[str, Any], edit_method: Dict[str, Any], edit_cfg: Dict[str, Any]) -> None:
    """Compute metrics and store result for the given configuration

    Args:
        metric (EditMetric): Metric object to use for computation.
        eval_dir (Path): Path to result directory. Must match path.
        path (str): Path to result directory. Must match eval_dir.
        data (str): Name of the dataset where eval.py was run on
        method (Dict[str, Any]): Inversion method
        edit_method (Dict[str, Any]): Editing method
        edit_cfg (Dict[str, Any]): Unused
    """

    if isinstance(metric, str):
        metric = EditMetric(metric, input_range=(-1, 1))
    
    enable_deterministic()
    path = eval_dir
    assert str(eval_dir) == str(path)

    # store results in metrics/{metric_name}.yaml under the respective result directory
    metric_out_file = eval_dir / "metrics" / f"{metric}.yaml"
    metric_out_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        # create empty file
        open(metric_out_file, 'x').close()
    except FileExistsError:
        # if file exists, metrics already have been computed -> skip
        print(f"skipping {metric_out_file}")
        return

    metric_res = []

    # load dataset for gt data

    if not isinstance(data, dict):
        data = {"type": data}

    data = {**data}
    data_name = data.pop("type")
    data_cfg = data
    # data = EditResultData(data_name, method, edit_method, path=path, skip_img_load=True, skip_existing=not override, **data_cfg)

    data = EditResultData(data_name, method, edit_method, path=path, skip_img_load=True, **data_cfg)

    # preprocessing of stored images
    preproc = StablePreprocess("cuda", size=512, center_crop=True)

    for i in range(len(data)):
        # iterate over dataset
        sample = data[i]

        # TODO: refactor
        # load source image (gt) from dataset
        image_file = sample.get("image_file", None)
        if image_file is None:
            source_image = sample["image"]
        else:
            source_image = Path(sample["image_file"])
            if not source_image.exists():
                source_image = None

        if source_image is not None and Path(sample["edit_image_file"]).exists():
            gc.collect()
            torch.cuda.empty_cache()
            # if source image and output image exists

            # some metrics need the edited word as argument -> grab from prompt-to-prompt config
            ptp_config = sample["edit"].get("ptp", None)
            blend_words = ptp_config.get("blend_words", []) if ptp_config is not None else []
            if isinstance(blend_words, (tuple, list)) and len(blend_words) == 2 and len(blend_words[0]) == 1:
                edit_word = blend_words[0][0]
            else:
                edit_word = None

            # add example to metric
            try:
                loss = metric.update(
                    source_image=preproc(source_image), 
                    edit_image=preproc(sample["edit_image_file"]), 
                    source_prompt=sample["source_prompt"], 
                    target_prompt=sample["edit"]["target_prompt"],
                    edit_word=edit_word,
                    mask=sample.get("mask", None))
            except Exception as e:
                loss = float("nan")
                print(f"Skipping {image_file} because of {e}")

            # record loss for each example
            metric_res.append({
                "value": loss,
                "file": Path(sample["edit_image_file"]).stem,
            })
        else:
            print(f"Skipping {sample['edit_image_file']}")

    # compute final metric value and store result
    metric_res = {
        "name": str(metric),
        "mean": metric.compute()[0],
        "results": metric_res,
    }

    metric_out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metric_out_file, "w") as f:
        yaml.dump(metric_res, f, Dumper=yaml.CSafeDumper)


@torch.no_grad()
def main(cfg: Optional[List[str]], metric: List[str], device: Optional[List[str]]) -> None:
    torch.multiprocessing.set_start_method('spawn')
    
    if metric is None:
        print("Using all metrics")
        metric = EditMetric.get_available_metrics()

    eval_dirs = []

    # create metrics
    metric = [EditMetric(m, input_range=(-1, 1)) for m in metric]

    for cfg_file in cfg:
        # for each config get all experiment directories
        cfg_name = Path(cfg_file).stem
        exp_dir = Path(f"result/{cfg_name}")  # get directory from config file name
        eval_dirs += sorted([d for d in exp_dir.glob("*/") if d.is_dir()])  # glob all experiments in the evaluation run

    jobs = []

    for eval_dir in tqdm(eval_dirs):
        # compute metrics for every experiemnt
        cfg = eval_dir / "cfg.yaml"
        with open(cfg, "r") as f:
            cfg = yaml.safe_load(f)
        
        for m in metric:
            jobs.append(dict(target=run_compute_metric, args=(m, eval_dir), kwargs=cfg))

    DevicePool(device).run(jobs)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", nargs="+", help="Config file(s) for evaluation.")
    parser.add_argument("--metric", nargs="+", help="Metric(s) to compute. If not specified, all metrics are computed.", 
                        choices=EditMetric.get_available_metrics(), metavar="")
    parser.add_argument("--device", nargs="+", help="Which cuda devices to use. Can be multiple (multiprocessing).")
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    main(**parse_args())
