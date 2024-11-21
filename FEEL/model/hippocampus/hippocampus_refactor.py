import torch
import torch.linalg
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler, DataLoader, Sampler
from typing import Literal, Generator, Tuple, Sequence, Any
from queue import PriorityQueue
import os
import json

from .database import VectorDatabase
from .event_dataset_refactor import EventDataset, EventData, event_data

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

class HippocampusRefactored():
    """A class representing the hippocampus of the brain
    - Memory loss occurs when replayed
    """
    # Episode and event
    dim_event: int
    event_per_episode: int
    min_event_for_episode: int
    
    # Replay
    min_event_for_replay: int
    replay_rate: int
    episode_per_replay: int
    times_replayed: int = 0
    
    # Data count
    max_len_dataset: int
    
    # Loss
    min_event_for_loss: int
    enable_loss: bool = True
    loss_freq: int
    loss_rate: float
    
    # Priority
    base_priority: float
    priority_method: tuple[Literal["base"], Literal["rate"]] = ("base", "rate")
    priority_queue: PriorityQueue
    
    # Generator
    id_generator: Counter
    replay_generator: Sampler = None
    data_loader: DataLoader = None
    
    stm: VectorDatabase
    event_dataset: EventDataset
    
    def __init__(
        self,
        dim_event: int = 768,
        event_per_episode: int = 3,
        min_event_for_episode: int = 8,
        replay_rate: int = 10,
        episode_per_replay: int = 5,
        min_event_for_replay: int = 100,
        max_len_dataset: int = 1000,
        min_event_for_loss: int = 100,
        loss_freq: int = 10,
        loss_rate: float = 0.1,
        base_priority: float = 100.,
    ):
        """_summary_

        Args:
            dim_event (int, optional): Dimension of event characteristics. Defaults to 768. (Note: unused)
            event_per_episode (int, optional): Number of episodes in an event, or event_size. Defaults to 3.
            min_event_for_episode (int, optional): Minimum number of event to generate an episode. Defaults to 8.
            replay_rate (int, optional): Replay rate. Defaults to 10. (Note: unused)
            episode_per_replay (int, optional): Number of episode to generate while replaying. Defaults to 5.
            min_event_for_replay (int, optional): Number of events to call replay, aka `replay_iteration`. Defaults to 100.
            max_len_dataset (int, optional): Maximum length of dataset. Defaults to 1000.
            min_event_for_loss (int, optional): Minimum number of events for memory loss to happen. Defaults to 100.
            loss_freq (int, optional): How often memory loss happens. Defaults to 10.
            loss_rate (float, optional): Ratio of memory to lose. Defaults to 0.1.
            base_priority (float, optional): Base priority. Defaults to 100..
        """
        self.dim_event = dim_event
        self.event_per_episode = event_per_episode
        self.min_event_for_episode = min_event_for_episode
        self.min_event_for_replay = min_event_for_replay
        self.replay_rate = replay_rate
        self.episode_per_replay = episode_per_replay
        self.max_len_dataset = max_len_dataset
        self.min_event_for_loss = min_event_for_loss
        self.loss_freq = loss_freq
        self.loss_rate = loss_rate
        self.base_priority = base_priority
        
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
            return torch.abs(eval1) + self.base_priority
        else:
            raise NotImplementedError(f"Method {method} for Hippocampus.init_priority() is not defined.")
    
    def update_priority(self, event_id: int, method: Literal["rate", "replace"] = "rate", new_eval2: torch.Tensor = None, rate: float = 1.0):
        if method == "replace" and new_eval2 is None:
            raise ValueError("new_eval2 must be provided when method is replace")
        self.event_dataset.update_priority(
            id=event_id,
            method=method,
            eval2=new_eval2 if method=="replace" else None,
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
            Tuple[int, torch.Tensor, float, torch.Tensor, float]: id, characteristics, eval1, eval2, priority
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
    
    def replay(self, batch_size: int = 1) -> EventData:
        """Replays memory
        1. Samples events from meory and outputs it according to the time of memory and impact
        2. Delete samples if necessary

        Args:
            batch_size (int, optional): Batch size. Defaults to 1.
        """
        events = self.sample(batch_size)
        if events is None:
            raise ValueError("Not enough events to replay")
        ids, characteristics, eval1s, eval2s, _ = events
        # Refresh priority
        for i, event_id in enumerate(ids):
            self.event_dataset.update_priority(event_id, self.priority_method[1], eval1=eval1s[i])
            self.times_replayed += 1
            if self.times_replayed % self.loss_freq == 0 or len(self) > self.max_len_dataset:
                self.organize_memory()
        return event_data(events)
    
    def generate_episode(self, event_id: int = None, event: EventData = None, characteristics: torch.Tensor = None) -> torch.Tensor:
        """Generates episodes from given ids

        Args:
            event_id (int, optional): The id of the event. Defaults to None.
            event (EventData, optional): The event. Defaults to None.
            characteristics (torch.Tensor, optional): The characteristics. Defaults to None.

        Raises:
            ValueError: If batch size does not match the number of events
            ValueError: If no matching event was found

        Returns:
            torch.Tensor: Episode of size [event_per_episode, dim_event]
        """
        if self.event_dataset.has_id(event_id):
            id, characteristics, *_, priority = self.get_event(event_id)
        elif event is not None:
            characteristics = event.characteristics
        elif characteristics is None:
            raise ValueError("No event or characteristics was found")
        
        episode = [characteristics]
        associated_id = []
        associated_priority = []
        result_list = self.search(
            k=self.event_per_episode-1,
            characteristics=characteristics
        )
        episode = [characteristics]
        for result in result_list:
            id, characteristics, *_, priority = self.get_event(result[0])
            episode.append(characteristics)
        for i, id in enumerate(associated_id):
            priority = associated_priority[i]
            self.event_dataset.update_priority(id, self.priority_method[1], eval1=priority, rate=0.5)
        return torch.stack(episode)
    
    def generate_episodes_batch(self, event_ids: Sequence[int] = None, events: Sequence[EventData] = None, characteristics: Sequence[torch.Tensor] = None, batch_size: int = None):
        """Generates episodes from given ids

        Args:
            event_ids (Sequence[int], optional): List of event ids. Defaults to None.
            events (Sequence[EventData], optional): List of events. Defaults to None.
            characteristics (Sequence[torch.Tensor], optional): The characteristics. Defaults to None.
            batch_size (int, optional): Batch size. Defaults to None.

        Raises:
            ValueError: If batch size does not match the number of events
            ValueError: If no matching event was found

        Returns:
            torch.Tensor: Episode of size [event_per_episode, dim_event]
        """
        if event_ids is not None:
            if batch_size is None:
                batch_size = len(event_ids)
            if batch_size != len(event_ids):
                raise ValueError(f"Batch size does not match the number of events: expected {batch_size}, got {len(event_ids)}")
            episodes = []
            for i in range(batch_size):
                episodes.append(self.generate_episode(event_id=event_ids[i]))
            return torch.stack(episodes)
        elif events is not None:
            if batch_size is None:
                batch_size = len(events)
            if batch_size != len(events):
                raise ValueError(f"Batch size does not match the number of events: expected {batch_size}, got {len(events)}")
            episodes = []
            for i in range(batch_size):
                episodes.append(self.generate_episode(event=events[i]))
            return torch.stack(episodes)
        elif characteristics is not None:
            if batch_size is None:
                batch_size = len(characteristics)
            if batch_size != len(characteristics):
                raise ValueError(f"Batch size does not match the number of events: expected {batch_size}, got {len(characteristics)}")
            episodes = []
            for i in range(batch_size):
                episodes.append(self.generate_episode(characteristics=characteristics[i]))
            return torch.stack(episodes)
        else:
            raise ValueError("No event or characteristics was found")
    
    def receive(self, characteristics: torch.Tensor, eval1: torch.Tensor, batch_size: int|None = None) -> list[EventData]:
        """Receives a batch of data and creates ids

        Args:
            characteristics (torch.Tensor): Characteristics of size [batch_size, dim_characteristics]
            eval1 (torch.Tensor): Characteristics of size [batch_size]
            batch_size (int | None, optional): Batch size. Defaults to None.

        Raises:
            TypeError: If shape of characteristics or eval1 is invalid

        Returns:
            list[EventData]: list of EventData objects
        """
        # Checks shape of characteristics and eval1
        if batch_size is None:
            batch_size = characteristics.size(0)
        if batch_size != characteristics.size(0):
            raise TypeError(f"Wrong size of characteristics: expected ({batch_size}, n), got {characteristics.size()}")
        if batch_size != eval1.size(0):
            raise TypeError(f"Wrong size of eval1: expected ({batch_size}), got {eval1.size()}")
        events = []
        for i in range(batch_size):
            events.append(EventData(next(self.id_generator), characteristics[i], eval1[i], None, None))
        return events
    
    def save_to_memory(
        self,
        event: EventData,
        event_id: int = None,
        characteristics: torch.Tensor = None,
        eval1: float = None,
        eval2: torch.Tensor = None,
        priority: float = None
    ):
        """Saves event to memory

        Args:
            event (EventData): Event to save
        """
        event_id = event_id if event_id is not None else event.id
        characteristics = characteristics if characteristics is not None else event.characteristics
        eval1 = eval1 if eval1 is not None else event.eval1
        eval2 = eval2 if eval2 is not None else event.eval2
        priority = priority if priority is not None else self.init_priority(event_id, eval1, self.priority_method[0])
        self.event_dataset.add_item(event_id, characteristics, eval1, eval2, priority)
        self.stm.add(event_id, characteristics)
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
            "min_event_for_episode": self.min_event_for_episode,
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
    def load_from_file(cls, file_path: str) -> "HippocampusRefactored":
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
        self.min_event_for_episode = dict_config["min_event_for_episode"]
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
        