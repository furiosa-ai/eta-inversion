

import random
from typing import Any, Dict, Type, List


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
    def __init__(self, dataset_cls: Type[DatasetBase], length: int=None, indices: List[int]=None, shuffle: bool=True, seed: int=0, **kwargs) -> None:
        """Generates a smaller subset of a dataset.

        Args:
            dataset (Type[DatasetBase]): Dataset class to create a subset of.
            length (Optional[int], optional): How many items to include (cuts of remaining items). Defaults to None. 
            indices (Optional[List[int]], optional): Indices to include. Don't specify length if you pass this argument. Defaults to None. 
            shuffle (bool, optional): If data should be shuffled before being cut off. Defaults to True. 
            seed (int, optional): Seed to shuffle data with. Defaults to 0. 
        """
        super().__init__()

        self.dataset = dataset_cls(**kwargs)
        self.length = length

        if indices is None:
            self.ind = list(range(len(self.dataset)))
        else:
            self.ind = indices
            self.length = len(indices)

        if shuffle:
            random.Random(seed).shuffle(self.ind)

    @property
    def skip_img_load(self):
        return self.dataset.skip_img_load

    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        return self.dataset[self.ind[idx]]

    def __repr__(self) -> str:
        return f"{self.dataset}_{self.length}"
