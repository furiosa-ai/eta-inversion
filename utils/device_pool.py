
from multiprocessing import Process, Queue
from tqdm import tqdm, trange
from queue import Queue, Empty
from threading import Thread, Lock
from typing import List, Dict
import os


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


class DevicePool:
    def __init__(self, devices) -> None:
        self.devices = devices if devices is not None else []

    def run(self, funcs: List[Dict]):
        procs = Queue()
        for func in funcs:
            procs.put(Process(**func))

        if len(self.devices) == 0:
            # for debugging don't use processes
            for _ in trange(procs.qsize()):
                proc = procs.get(block=False)
                proc._target(*proc._args, **proc._kwargs)
        else:
            # create consumer threads for each device which will consume the processes created above
            pbar = tqdm(total=procs.qsize())
            lk = Lock()
            workers = [Thread(target=worker_func, args=(device, procs, lk, pbar)) for device in self.devices]
            
            # start and join consumers
            for worker in workers:
                worker.start()

            for worker in workers:
                worker.join()

            pbar.close()
