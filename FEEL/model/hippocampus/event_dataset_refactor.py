import torch
from torch.utils.data import Dataset
from typing import Literal, Any, TypeVar, Tuple, Callable
from dataclasses import dataclass
import pandas as pd
import numpy as np
from functools import wraps

T = TypeVar("T")

def compress0d(obj_: int|float|list|torch.Tensor|np.ndarray) -> int|float:
    """Compresses 1-element list-like object (ndarray, Tensor, list) to scalar

    Args:
        obj_ (int | float | list | torch.Tensor | np.ndarray): Object to compress

    Raises:
        ValueError: If obj_ cannot be compressed

    Returns:
        int|float: Compressed scalar
    """
    if isinstance(obj_, int|float):
        return obj_
    elif isinstance(obj_, list):
        if len(obj_) == 1:
            return compress0d(obj_[0])
        else:
            raise ValueError(f"Invalid length {len(obj_)}")
    elif isinstance(obj_, torch.Tensor|np.ndarray):
        return obj_.item()

@dataclass
class EventData:
    """A simple wrapper for data acquired by EventDataset
    """
    id: int
    characteristics: torch.Tensor
    eval1: float
    eval2: torch.Tensor
    priority: float

def _parse_event_data(fn: Callable[T, Tuple[int, torch.Tensor, float, torch.Tensor, float]]) -> Callable[T, EventData]:
    @wraps(fn)
    def wrapper(*args, **kwargs):
        res = fn(*args, **kwargs)
        if res is not None:
            return EventData(*res)
        return None
    return wrapper

def parse_event_data(obj_: Any) -> EventData:
    if isinstance(obj_, tuple):
        return EventData(*obj_)
    elif isinstance(obj_, EventData):
        return obj_
    else:
        raise TypeError(f"Unsupported type {type(obj_)} for parse_event_data()")

class EventDataset(Dataset):
    """Dataset class for storing memory

    Args:
        data (torch.Tensor): Characteristic of event (video)
        eval1 (float): Intuitive emotional response
        eval2 (torch.Tensor): Emotional response handled in Prefrontal Cortex
        priority (float): Priority
    """
    _df: pd.DataFrame
    
    def __init__(
        self
    ):
        self._df = pd.DataFrame(
            data=[],
            columns=["characteristics", "eval1", "eval2", "priority"]
        )
        self._cast_type()
    
    def __len__(self):
        return len(self._df.index)
    
    # @parse_event_data
    def __getitem__(self, idx: int) -> Tuple[int, torch.Tensor, float, torch.Tensor, float]:
        """Gets from dataset

        Args:
            idx (int): Index

        Returns:
            Tuple[int, torch.Tensor, float, torch.Tensor, float]: id, characteristics, eval1, eval2, priority
        """
        row = self._df.iloc[idx]
        return row.name, row["characteristics"], row["eval1"], row["eval2"], row["priority"]
    
    def get_priority(self) -> list:
        """Returns a tensor of priority

        Returns:
            torch.Tensor: Tensor containing priority of shape [len(dataset)]
        """
        return self._df["priority"].tolist()
    
    def has_id(self, id: int) -> bool:
        return id in self._df.index
    
    # @parse_event_data
    def get_by_id(self, id: int) -> Tuple[int, torch.Tensor, float, torch.Tensor, float]:
        """Gets data from unique id

        Args:
            id (int): id
        
        Raises:
            ValueError: If id does not exist
        
        Returns:
            Tuple[int, torch.Tensor, float, torch.Tensor, float]: id, characteristics, eval1, eval2, priority
        """
        if not self.has_id(id):
            raise ValueError(f"Data with id {id} does not exist")
        row = self._df.loc[id]
        return id, row["characteristics"], row["eval1"], row["eval2"], row["priority"]
    
    def remove_by_id(self, id: int) -> None:
        """Removes data with the specified id

        Args:
            id (int): id to remove
            
        Raises:
            ValueError: If id does not exist
        """
        if not self.has_id(id):
            raise ValueError(f"Data with id {id} does not exist")
        del self._df.loc[id]
    
    def add_item(self, id: int, characteristics: torch.Tensor, eval1: Any, eval2: torch.Tensor, priority: Any) -> None:
        """Adds item to dataset

        Args:
            id (int): id
            characteristics (torch.Tensor): Characteristic of video
            eval1 (Any): Intuitive emotional response, either in float, Tensor or ndarray
            eval2 (torch.Tensor): Emotional response after processing
            priority (Any): Priority, either in float, Tensor or ndarray
        """
        assert not self.has_id(id), ValueError(f"Data with id {id} already exists")
        eval1 = compress0d(eval1)
        priority = compress0d(priority)
        self._df.loc[id] = {"characteristics": characteristics, "eval1": eval1, "eval2": eval2, "priority": priority}
    
    def update_priority(self, id: int, method: Literal["rate", "replace"], eval1: float, rate: float=1.0) -> None:
        assert self.has_id(id), f"Data with id {id} does not exist"
        if method == "rate":
            self._df.at[id, "priority"] += rate * self._df.at[id, "eval1"]
        elif method == "replace":
            self._df.at[id, "priority"] = eval1
        else:
            raise TypeError(f"Unsupported method for update_priority(): {method}")
    
    def save_to_file(self, file_path: str) -> None:
        """Save dataset to file

        Args:
            file_path (str): File path to save to
        """
        def handler(input: Any) -> Any:
            if isinstance(input, torch.Tensor):
                return input.tolist()
            elif isinstance(input, np.ndarray):
                return input.tolist()
            else:
                raise TypeError(f"Unsupported parsing type {type(input)}")
        self._df.to_json(file_path, default_handler=handler)
    
    def _cast_type(self) -> None:
        self._df["characteristics"] = self._df["characteristics"].astype(object).apply(torch.Tensor)
        self._df["eval1"] = self._df["eval1"].apply(compress0d).astype(float)
        self._df["eval2"] = self._df["eval2"].astype(object).apply(torch.Tensor)
        self._df["priority"] = self._df["priority"].apply(compress0d).astype(float)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> "EventDataset":
        """Load EventDataset from file_path

        Args:
            file_path (str): File path to load
        """
        self = cls()
        self._df = pd.read_json(file_path)
        self._cast_type()
        return self