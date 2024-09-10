import json
import pathlib
import random
import numpy as np
import torch
from typing import Union
from rich.console import Console

def save_json(data, path: Union[str, pathlib.Path]):
    if isinstance(path, str):
        path = pathlib.Path(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    if path.suffix != '.json':
        path = path.with_suffix('.json')
    with open(path, 'w') as f:
        json.dump(data, f, indent=4, sort_keys=False)

def load_json(path: Union[str, pathlib.Path]):
    if isinstance(path, str):
        path = pathlib.Path(path)
    if path.suffix != '.json':
        path = path.with_suffix('.json')
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class FileLoggingConsole(Console):
    def __init__(self, path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = path

    def log(self, *args, **kwargs):
        super().log(*args, **kwargs)
        with open(self.path, "a") as f:
            f.write(self.export_text())
        self.clear()