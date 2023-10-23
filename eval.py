
from utils.debug_utils import enable_deterministic
enable_deterministic()

from pathlib import Path
from typing import Optional, Dict, Any
import cv2
import argparse
from tqdm import trange, tqdm
import yaml
import torch
import os

from modules import load_inverter, load_editor
# from modules.exceptions import DiffusionInversionException
from utils.eval_utils import EditResultData, create_configs
from modules import StablePreprocess, StablePostProc
import time
from diffusers import StableDiffusionPipeline
from multiprocessing import Process
from threading import Thread, Lock
from queue import Queue, Empty
from typing import List, Optional


def run_eval(path: str, data: str, method: Dict[str, Any], edit_method: Dict[str, Any], edit_cfg: Dict[str, Any]) -> None:
    """Edits all images in the dataset with the given configuation and stores all output images

    Args:
        path (str): Path to save output images
        data (str): Name of the dataset
        method (Dict[str, Any]): Inversion method configuration
        edit_method (Dict[str, Any]): Editing method configuration
        edit_cfg (Dict[str, Any]): Unused
    """

    enable_deterministic()
    device = "cuda"
    # metric_name = metric

    # Loads and manages dataset for evaluation
    data = EditResultData(data, method, edit_method, path=path, skip_img_load=True)

    # load diffusion model
    ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)
    preproc = StablePreprocess(device, size=512, center_crop=True, return_np=False, pil_resize=True)
    postproc = StablePostProc()

    # load inverter and editor module
    inverter = load_inverter(model=ldm_stable, **method)
    editor = load_editor(inverter=inverter, **edit_method)

    for i in trange(len(data)):
        # iterate over evaluation dataset
        sample = data[i]

        # needs refactoring
        image_file = sample.get("image_file", None)
        image = preproc(image_file if image_file is not None else sample["image"])
        source_prompt = sample["source_prompt"]
        target_prompt = sample["edit"]["target_prompt"]

        # get editing config for the current example, if exists (necessary for ptp)
        edit_cfg = sample["edit"].get(edit_method["type"], None)

        res = editor.edit(image, source_prompt, target_prompt, edit_cfg)

        if res is not None:
            # if successfully edited
            edit_image = postproc(res["image"])
            Path(sample["edit_image_file"]).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(sample["edit_image_file"]), cv2.cvtColor(edit_image, cv2.COLOR_RGB2BGR))
        else:
            # failed editing. might be an incompatible combination of inverter and editor or an unimplemented feature.
            pass


def worker_func(device: str, procs: Queue[Process], lk: Lock, pbar: tqdm) -> None:
    """Consumer thread which will fetch processes

    Args:
        device (str): Device assigned to this consumer
        procs (Queue[Process]): Global process queue
        lk (Lock): Global lock
        pbar (tqdm): For displaying progress
    """
    while True:
        # avoid that other processes get started
        with lk:
            try:
                proc = procs.get(block=False)
            except Empty:
                return

            # set device for process
            os.environ["CUDA_VISIBLE_DEVICES"] = device
            proc.start()
        proc.join()
        pbar.update(1)


@torch.no_grad()
def main(cfg: List[str], device: Optional[List[str]], no_proc: bool) -> None:
    # necessary for multiprocessing
    torch.multiprocessing.set_start_method('spawn')

    # create one process job per eval configuration
    procs = Queue()
    eval_idx = 0

    for cfg_file in cfg:
        # for each config file create all possible combinations of dataset, editing method, inversion method, ...
        cfgs, cfg_all = create_configs(cfg_file)

        # create result path
        Path(cfg_all["path"]).mkdir(parents=True, exist_ok=True)

        # dump global config
        with open(Path(cfg_all["path"]) / "cfg.yaml", "w") as f:
            yaml.dump(cfg_all, f, Dumper=yaml.CSafeDumper)

        for i, cfg in enumerate(cfgs):
            # for each combination of dataset, editing method and inversion method

            # dump config
            Path(cfg["path"]).mkdir(parents=True, exist_ok=True)
            with open(Path(cfg["path"]) / "cfg.yaml", "w") as f:
                yaml.dump(cfg, f, Dumper=yaml.CSafeDumper)

            # create process
            procs.put(Process(target=run_eval, kwargs=cfg))
            eval_idx += 1

    # select device to use
    devices = device if device is not None else [os.environ.get("CUDA_VISIBLE_DEVICES", "0")]

    if no_proc:
        # for debugging don't use processes
        for _ in trange(procs.qsize()):
            proc = procs.get(block=False)
            proc._target(**proc._kwargs)
    else:
        # create consumer threads for each device which will consume the processes created above
        pbar = tqdm(total=procs.qsize())
        lk = Lock()
        workers = [Thread(target=worker_func, args=(device, procs, lk, pbar)) for device in devices]
        
        # start and join consumers
        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()

        pbar.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation for the given config file. The result will be stored under result/{cfg_file_name}. " \
                                     "For each combination of dataset, inversion and editing method in the config file, a separate directory will be created in result/{cfg_file_name}")
    parser.add_argument("--cfg", required=True, nargs="+", help="Config file(s) for evaluation.")
    parser.add_argument("--device", nargs="+", help="Which cuda devices to use. Can be multiple (multiprocessing).")
    parser.add_argument("--no_proc", action="store_true", help="Disables multiprocessing.")
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    main(**parse_args())
