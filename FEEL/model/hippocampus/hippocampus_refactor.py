import torch
import torch.linalg
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler, DataLoader
from typing import Literal, Any, Generator, Tuple
from queue import PriorityQueue
import os
import json

from .database import VectorDatabase
from .event_dataset_refactor import EventDataset, EventData, parse_event_data, compress0d

class Counter():
    """A class representing a counter
    """
    i: int
    def __init__(self, start: int = 0):
        self.i = start
    
    def __iter__(self):
        return self
    
    def __next__(self) -> int:
        self.i += 1
        return self.i

class Hippocampus():
    """A class representing the hippocampus of the brain
    """
    # Episode and event
    dim_event: int
    event_per_episode: int
    
    # Replay
    min_event_for_replay: int = 100
    replay_rate: int
    episode_per_replay: int
    times_replayed: int = 0
    
    # Data count
    max_len_dataset: int = 1000
    
    # Loss
    min_event_for_loss: int = 100
    enable_loss: bool = True
    loss_freq: int = 10
    loss_rate: float = 0.1
    
    # Priority
    base_priority: float = 100.
    priority_method: tuple[Literal["base"], Literal["rate"]] = ("base", "rate")
    priority_queue: PriorityQueue
    
    # Generator
    id_generator: Counter
    replay_generator: Any = None
    data_loader: DataLoader = None
    
    stm: VectorDatabase
    event_dataset: EventDataset
    
    def __init__(
        self,
        dim_event: int = 768,
        event_per_episode: int = 3,
        replay_rate: int = 10,
        episode_per_replay: int = 5
    ):
        self.dim_event = dim_event
        self.event_per_episode = event_per_episode
        self.replay_rate = replay_rate
        self.episode_per_replay = episode_per_replay
        self.priority_queue = PriorityQueue()
        self.stm = VectorDatabase(dim_event)
        self.event_dataset = EventDataset()
        self.id_generator = Counter()
    
    def __len__(self) -> int:
        return len(self.event_dataset)
    
    def get_ids(self) -> list[int]:
        return self.event_dataset._df.index.tolist()
    
    def init_priority(self, event_id: int, eval1: torch.Tensor|float, method: Literal["base"] = "base"):
        if method == "base":
            return compress0d(eval1) + self.base_priority
        else:
            raise NotImplementedError(f"Method {method} for Hippocampus.init_priority() is not defined.")
    
    def update_priority(self, event_id: int, new_eval: torch.Tensor, method: Literal["rate", "replace"] = "rate", rate: float = 1.0):
        self.event_dataset.update_priority(
            id=event_id,
            method=method,
            eval2=new_eval,
            rate=rate
        )
    
    def sample(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Generates a sample from stored data

        Args:
            batch_size (int, optional): Size of batch. Defaults to 1.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing id, data, eval1, eval2, priority
        """
        if len(self) <= self.min_event_for_replay:
            return None
        if self.replay_generator is None or self.times_replayed % self.loss_freq == 0:
            # Refresh weight randomly
            weights = self.event_dataset.get_priority()
            self.replay_generator = WeightedRandomSampler(weights, self.episode_per_replay)
            self.data_loader = DataLoader(self.event_dataset, batch_size, sampler=self.replay_generator)
        return next(iter(self.data_loader))
    
    def get_event(self, event_id: int) -> Tuple[int, torch.Tensor, float, torch.Tensor, float]:
        """Gets an event by id

        Args:
            event_id (int): The id of an event

        Returns:
            Tuple[int, torch.Tensor, float, torch.Tensor, float]: id, data, eval1, eval2, priority
        """
        return self.event_dataset.get_by_id(event_id)
    
    def search(self, k: int = 5, event_id: int = -1, characteristics: torch.Tensor|None = None) -> list[Tuple[int, float]]:
        """Get a memory similar to that of the given event id

        Args:
            k (int, optional): How many events to search for. Defaults to 5.
            event_id (int, optional): The id of the event. Defaults to -1.
            characteristics (torch.Tensor | None, optional): The characteristics to look for. Defaults to None.
        
        Returns:
            list[Tuple[int, float]]: List of tuples containing id and distance
        """
        if characteristics is None:
            assert self.event_dataset.has_id(event_id), "No matching event was found"
            characteristics = self.get_event(event_id)[1]
        return self.stm.search(characteristics, k)

    def organize_memory(self) -> None:
        """Deletes some memory
        """
        # Only delete if enable_loss and dataset length exceeds min_event_for_loss
        if not self.enable_loss or len(self) < self.min_event_for_loss:
            return
        num_removed = int((len(self)-self.min_event_for_loss)*self.loss_rate)
        memories_removed = []
        for i in range(num_removed):
            id_removed, _ = self.priority_queue.get()
            # Removes from priority queue, dataset, and short term memory (vector database)
            self.event_dataset.remove_by_id(id_removed)
            memories_removed.append(id_removed)
            self.stm.remove(id_removed)
    
    def replay(self, batch_size: int = 1):
        """Replays memory
        1. Samples events from meory and outputs it according to the time of memory and impact
        2. Delete samples if necessary

        Args:
            batch_size (int, optional): Batch size. Defaults to 1.
        """
        events = self.sample(batch_size)
        assert events is not None, "No events to replay"
        ids = events[0]
        # Refresh priority
        for event_id in ids:
            self.event_dataset.update_priority(event_id, self.priority_method[1])
            self.times_replayed += 1
            if self.times_replayed % self.loss_freq == 0:
                self.organize_memory()
        return events
    
    def generate_episode(self, event_id: int, batch_size: int = 1) -> torch.Tensor:
        """Searches event relevant to the given event and returns an episode

        Args:
            event (_type_): Event to search the related memory for
            batch_size (int, optional): Batch size. Defaults to 1.

        Returns:
            torch.Tensor: Episode of size [batch_size, event_per_episode, dim_event]
        """
        if len(self) <= self.min_event_for_replay:
            return None
        if not self.event_dataset.has_id(event_id):
            raise ValueError("No matching event was found")
        event = parse_event_data(self.get_event(event_id))
        episode_batch = []
        for i in range(batch_size):
            result_list = self.search(
                k=self.event_per_episode-1,
                characteristics=event.data
            )
            episode = [event.data]
            for result in result_list:
                _, data, *_ = self.get_event(result[0])
                episode.append(data)
            episode = torch.stack(episode)
            episode_batch.append(episode)
        return torch.stack(episode_batch)
    
    def receive(self, data: torch.Tensor, eval1: float) -> EventData:
        """Gets new data and stores it

        Args:
            data (torch.Tensor): _description_
            eval1 (float): _description_
        """
        event_id = next(self.id_generator)
        event = EventData(event_id, data, eval1, None, None)
        return event
    
    def save_to_memory(
        self,
        event: EventData,
        event_id: int = -1,
        data: torch.Tensor = None,
        eval1: float = None,
        eval2: torch.Tensor = None,
        priority: float = None
    ):
        """Saves event to memory

        Args:
            event (EventData): Event to save
        """
        event_id = event_id if event_id != -1 else event.id
        data = data if data is not None else event.data
        eval1 = eval1 if eval1 is not None else event.eval1
        eval2 = eval2 if eval2 is not None else event.eval2
        priority = priority if priority is not None else self.init_priority(event_id, eval1, self.priority_method[0])
        self.event_dataset.add_item(event_id, data, eval1, eval2, priority)
        self.stm.add(event_id, data)
        self.priority_queue.put((event_id, priority))
    
    def save_to_file(self, file_path: str) -> None:
        """Saves memory to file

        Args:
            file_path (str): Path to save file
        """
        base, _ = os.path.splitext(file_path)
        path_dataset = f"{base}_dataset.json"
        self.event_dataset.save_to_file(path_dataset)
        path_config = f"{base}_config.json"
        dict_config = {
            "dim_event": self.dim_event,
            "event_per_episode": self.event_per_episode,
            "min_event_for_replay": self.min_event_for_replay,
            "replay_rate": self.replay_rate,
            "episode_per_replay": self.episode_per_replay,
            "times_replayed": self.times_replayed,
            "max_len_dataset": self.max_len_dataset,
            "min_event_for_loss": self.min_event_for_loss,
            "enable_loss": self.enable_loss,
            "loss_freq": self.loss_freq,
            "loss_rate": self.loss_rate,
            "base_priority": self.base_priority,
            "priority_method": self.priority_method,
            "counter_current": self.id_generator.i
        }
        
        with open(path_config, "w") as f:
            json.dump(dict_config, f)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> "Hippocampus":
        """Loads and structures Hippocampus from file

        Args:
            file_path (str): File path

        Returns:
            Hippocampus: initialized object
        """
        base, _ = os.path.splitext(file_path)
        path_dataset = f"{base}_dataset.json"
        path_config = f"{base}_config.json"
        self = cls()
        with open(path_config, "r") as f:
            dict_config: dict = json.load(f)
        self.dim_event = dict_config["dim_event"]
        self.event_per_episode = dict_config["event_per_episode"]
        self.min_event_for_replay = dict_config["min_event_for_replay"]
        self.replay_rate = dict_config["replay_rate"]
        self.episode_per_replay = dict_config["episode_per_replay"]
        self.times_replayed = dict_config["times_replayed"]
        self.max_len_dataset = dict_config["max_len_dataset"]
        self.min_event_for_loss = dict_config["min_event_for_loss"]
        self.enable_loss = dict_config["enable_loss"]
        self.loss_freq = dict_config["loss_freq"]
        self.loss_rate = dict_config["loss_rate"]
        self.base_priority = dict_config["base_priority"]
        self.priority_method = dict_config["priority_method"]
        self.id_generator = Counter(dict_config["counter_current"])
        self.event_dataset = EventDataset.load_from_file(path_dataset)
        self.stm = VectorDatabase(self.dim_event)
        self.priority_queue = PriorityQueue()
        for i in range(len(self.event_dataset)):
            id, data, eval1, eval2, priority = self.event_dataset[i]
            self.stm.add(id, data)
            self.priority_queue.put((id, priority))
        return self
        