

import random
from typing import Any, Dict


class DatasetBase:
    """Base class for all datasets for evaluation
    """
    def __init__(self) -> None:
        self._iter_idx = None

    def __iter__(self):
        self._iter_idx = 0
        return self
    
    def __next__(self):
        if self._iter_idx >= len(self):
            self._iter_idx = None
            raise StopIteration

        x = self.__getitem__(self._iter_idx)
        self._iter_idx += 1
        return x


class DatasetSubset(DatasetBase):
    """A subset of dataset (mainly for debugging purposes)
    """
    def __init__(self, dataset: DatasetBase, length: int, shuffle: bool=True, seed: int=0) -> None:
        """Generates a smaller subset of a dataset.

        Args:
            dataset (DatasetBase): Dataset to make a subset of
            length (int): How many items to include (cuts of remaining items)
            shuffle (bool, optional): _description_. Defaults to True. If data should be shuffled before being cut off
            seed (int, optional): _description_. Defaults to 0. Seed to shuffle data with
        """
        super().__init__()

        self.dataset = dataset
        self.length = length
        self.ind = list(range(len(dataset)))

        if shuffle:
            random.Random(seed).shuffle(self.ind)

    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        return self.dataset[self.ind[idx]]
