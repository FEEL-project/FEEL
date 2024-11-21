import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle

class EventDataset(Dataset):
    def __init__(self, ids=None,  characteristics=None, evaluation1s=None, evaluation2s=None, priority=None):
        # Initialize with empty lists if no data provided
        self.ids = ids if ids is not None else []
        self.characteristics = characteristics if characteristics is not None else []
        self.evaluation1s = evaluation1s if evaluation1s is not None else []
        self.evaluation2s = evaluation2s if evaluation2s is not None else []
        self.priority = priority if priority is not None else []
        self._id_to_index = None
        
        # Validate input consistency
        if not (len(self.characteristics) == len(self.ids) == len(self.evaluation1s) == len(self.evaluation2s) == len(self.priority)):
            raise ValueError("characteristics, IDs, and evaluations, priority must have the same length")
        
        # Initialize id to index mapping
        self._update_id_to_index_mapping()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """return a single item by index"""
        # print(f"self.ids[idx]:{self.ids[idx]}")
        # print(f"self.characteristics[idx]:{self.characteristics[idx]}")
        # print(f"self.evaluation1s[idx]:{self.evaluation1s[idx]}")
        # print(f"self.evaluation2s[idx]:{self.evaluation2s[idx]}")
        # print("size:", len(self.ids), len(self.characteristics), len(self.evaluation1s), len(self.evaluation2s))
        return {
            'id': self.ids[idx], 
            'characteristics': torch.tensor(self.characteristics[idx]).float(),
            'evaluation1': torch.tensor(self.evaluation1s[idx]).float(),
            'evaluation2': torch.tensor(self.evaluation2s[idx]).float()
        }
        
    def get_priority(self) -> torch.Tensor:
        #  Return the priority array as a tensor
        return torch.tensor(self.priority).float()

    def _update_id_to_index_mapping(self):
        # Rebuild the id to index mapping (for thread safety)
        if self._id_to_index is not None:
            self._id_to_index.clear()
        self._id_to_index = {id_val: idx for idx, id_val in enumerate(self.ids)}

    def get_by_id(self, id_value):
        """Return an item by ID"""
        # {'id': self.ids[idx], 
        #  'characteristics': torch.from_numpy(self.characteristics[idx]).float(), 
        #  'evaluation': torch.from_numpy(self.evaluations[idx]).float()
        # }
        index = self._id_to_index.get(id_value)
        return self[index] if index is not None else None

    def add_item(self, new_id:int, new_characteristics: torch.Tensor, new_evaluation1: torch.Tensor, new_evaluation2: torch.Tensor, new_priority):
        """Add a single item to the dataset"""
        # Ensure new_characteristics has the same shape as the existing characteristics[0]
        if len(self.characteristics) > 0 and new_characteristics.shape != self.characteristics[0].shape:
            raise ValueError(f"characteristics shape {new_characteristics.shape} does not match existing characteristics shape {self.characteristics[0].shape}")
        
        # Check for duplicate ID
        if new_id in self._id_to_index:
            raise ValueError(f"ID {new_id} already exists in the dataset")
        
        # Append new characteristics and ID
        self.ids.append(new_id)
        self.characteristics.append(new_characteristics)
        self.evaluation1s.append(new_evaluation1)
        self.evaluation2s.append(new_evaluation2)
        self.priority.append(new_priority)
        
        # Update the index mapping
        self._update_id_to_index_mapping()

    def remove_by_id(self, id_to_remove):
        """Remove an item from the dataset by ID"""
        # Find the index of the ID to remove
        index = self._id_to_index.get(id_to_remove)
        
        if index is None:
            raise ValueError(f"ID {id_to_remove} not found in the dataset")
        
        # Remove the item at the found index
        del self.characteristics[index] # self.characteristics = np.delete(self.characteristics, index)
        del self.ids[index] # self.ids = np.delete(self.ids, index)
        del self.evaluation1s[index] # self.evaluation1s = np.delete(self.evaluation1s, index)
        del self.evaluation2s[index] # self.evaluation2s = np.delete(self.evaluation2s, index)
        del self.priority[index] # self.priority = np.delete(self.priority, index)
        
        # Rebuild the index mapping
        self._update_id_to_index_mapping()
        
    def update_priority(self, id_value:int, method:str, evaluation2=None, rate=1.0):
        """update the priority of an item by ID"""
        index = self._id_to_index.get(id_value)
        if index is None:
            print(f"EventDataset: {self._id_to_index}")
            raise ValueError(f"ID {id_value} (index {index}) not found in the dataset")
        ### issue: 要検討(アルゴリズム)
        if method == 'rate':
            """8次元の評価値をノルムにして、rateをかけてpriorityに加算"""
            self.priority[index] += rate * torch.norm(self.evaluation2s[index])
        elif method == 'replace':
            self.priority[index] = evaluation2
        else:
            raise ValueError(f"Invalid method: {method}")
        ### アルゴリズムは追加可能

    def bulk_add_items(self, new_characteristics_list, new_ids, new_evaluation1s, new_evaluation2s, new_priority):
        """Add multiple items to the dataset"""
        # Validate input lengths match
        if not (len(new_characteristics_list) == len(new_ids) == len(new_evaluation1s)):
            raise ValueError("Number of characteristics items must match number of IDs")
        
        # Check for duplicate IDs
        duplicate_ids = set(new_ids) & set(self.ids)
        if duplicate_ids:
            raise ValueError(f"Duplicate IDs found: {duplicate_ids}")
        
        # Append new characteristics and IDs
        self.characteristics.extend(new_characteristics_list)
        self.ids.extend(new_ids)
        self.evaluation1s.extend(new_evaluation1s)
        self.evaluation2s.extend(new_evaluation2s)
        self.priority.extend(new_priority)
        
        # Update the index mapping
        self._update_id_to_index_mapping()
    
    def bulk_delete_items(self, ids_to_remove):
        """Remove multiple items from the dataset by ID"""
        # Validate input
        if not ids_to_remove:
            return

        # Find indices to remove
        indices_to_remove = [self._id_to_index[id_val] for id_val in ids_to_remove if id_val in self._id_to_index]
        
        if not indices_to_remove:
            raise ValueError("None of the provided IDs were found in the dataset")

        # Sort indices in reverse to avoid shifting index issues during deletion
        indices_to_remove.sort(reverse=True)
        
        # Remove items at specified indices
        for idx in indices_to_remove:
            del self.characteristics[idx] # self.characteristics = np.delete(self.characteristics, idx)
            del self.ids[idx] # self.ids = np.delete(self.ids, idx)
            del self.evaluation1s[idx] # self.evaluation1s = np.delete(self.evaluation1s, idx)
            del self.evaluation2s[idx] # self.evaluation2s = np.delete(self.evaluation2s, idx)
            del self.priority[idx] # self.priority = np.delete(self.priority, idx)
        
        # Update the index mapping
        self._update_id_to_index_mapping()
        
    def save_to_file(self, file_path):
        # save EventDataset instance to file_path
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)
    
    @staticmethod
    def load_from_file(file_path):
        # load EventDataset instance from file_path
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
            if not isinstance(obj, EventDataset):
                raise ValueError(f"File does not contain an EventDataset instance but a {type(obj)}")
            return obj