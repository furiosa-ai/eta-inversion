from collections import OrderedDict
import torch
import numpy as np
import random


def _to_str(x):
    if isinstance(x, list):
        return "[" + ", ".join(_to_str(v) for v in x) + "]"
    elif isinstance(x, tuple):
        return "(" + ", ".join(_to_str(v) for v in x) + ")"
    elif isinstance(x, (dict, OrderedDict)):
        return "{" + ", ".join(f"{_to_str(k)}={_to_str(v)}" for k, v in x.items()) + "}"
    elif isinstance(x, np.ndarray):
        return str(np.mean(x))
    elif isinstance(x, torch.Tensor):
        return str(torch.mean(x, dtype=torch.float32).detach().cpu().numpy().item())
    elif isinstance(x, (str, int, float)):
        return str(x)
    else:
        return "<obj>"


def log_func_inputs(f):
    def _f(*args, **kwargs):
        f_code = f.__code__
        f_name = f_code.co_name

        kwargs = OrderedDict(sorted(kwargs.items()))

        print(f"{f_name}", _to_str(args), _to_str(kwargs))

        return f(*args, **kwargs)
    return _f


def enable_deterministic():
    """Does not work for every operator like backwards in NTI
    """
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
