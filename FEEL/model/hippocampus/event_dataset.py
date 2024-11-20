import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle

class EventDataset(Dataset):
    def __init__(self, ids=None,  data=None, evaluation1s=None, evaluation2s=None, priority=None):
        # Initialize with empty lists if no data provided
        self.ids = np.array(ids) if ids is not None else np.empty((0,), dtype=object)
        self.data = np.array(data) if data is not None else np.empty((0,), dtype=object)
        self.evaluation1s = np.array(evaluation1s) if evaluation1s is not None else np.empty((0,), dtype=object)
        self.evaluation2s = np.array(evaluation2s) if evaluation2s is not None else np.empty((0,), dtype=object)
        self.priority = np.array(priority) if priority is not None else np.empty((0,), dtype=object)
        self._id_to_index = None
        
        # Validate input consistency
        if not (len(self.data) == len(self.ids) == len(self.evaluation1s) == len(self.evaluation2s) == len(self.priority)):
            raise ValueError("Data, IDs, and evaluations, priority must have the same length")
        
        # Initialize id to index mapping
        self._update_id_to_index_mapping()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {
            'id': self.ids[idx], 
            'data': torch.from_numpy(self.data[idx]).float(),
            'evaluation1': torch.from_numpy(self.evaluation1s[idx]).float(),
            'evaluation2': torch.from_numpy(self.evaluation2s[idx]).float()
        }
        
    def get_priority(self) -> torch.Tensor:
        #  Return the priority array as a tensor
        return torch.from_numpy(self.priority).float()

    def _update_id_to_index_mapping(self):
        # Rebuild the id to index mapping (for thread safety)
        if self._id_to_index is not None:
            self._id_to_index.clear()
        self._id_to_index = {id_val: idx for idx, id_val in enumerate(self.ids)}

    def get_by_id(self, id_value):
        # {'id': self.ids[idx], 
        #  'data': torch.from_numpy(self.data[idx]).float(), 
        #  'evaluation': torch.from_numpy(self.evaluations[idx]).float()
        # }
        index = self._id_to_index.get(id_value)
        return self[index] if index is not None else None

    def add_item(self, new_id, new_data, new_evaluation1, new_evaluation2, new_priority):
        # Ensure new_data is a numpy array
        if not isinstance(new_data, np.ndarray):
            new_data = np.array(new_data)
        
        # Check for duplicate ID
        if new_id in self._id_to_index:
            raise ValueError(f"ID {new_id} already exists in the dataset")
        
        # Append new data and ID
        self.ids = np.append(self.ids, new_id)
        self.data = np.append(self.data, new_data)
        self.evaluation1s = np.append(self.evaluation1s, new_evaluation1)
        self.evaluation2s = np.append(self.evaluation2s, new_evaluation2)
        self.priority = np.append(self.priority, new_priority)
        
        # Update the index mapping
        self._update_id_to_index_mapping()

    def remove_by_id(self, id_to_remove):
        # Find the index of the ID to remove
        index = self._id_to_index.get(id_to_remove)
        
        if index is None:
            raise ValueError(f"ID {id_to_remove} not found in the dataset")
        
        # Remove the item at the found index
        self.data = np.delete(self.data, index)
        self.ids = np.delete(self.ids, index)
        self.evaluation1s = np.delete(self.evaluation1s, index)
        self.evaluation2s = np.delete(self.evaluation2s, index)
        self.priority = np.delete(self.priority, index)
        
        # Rebuild the index mapping
        self._update_id_to_index_mapping()
        
    def update_priority(self, id_value, method:str, evaluation2=None, rate=1.0):
        index = self._id_to_index.get(id_value)
        if index is None:
            raise ValueError(f"ID {id_value} not found in the dataset")
        ### issue: 要検討(アルゴリズム)
        if method == 'rate':
            self.priority[index] += rate * self.evaluation2s[index]
        elif method == 'replace':
            self.priority[index] = evaluation2
        else:
            raise ValueError(f"Invalid method: {method}")
        ### アルゴリズムは追加可能

    def bulk_add_items(self, new_data_list, new_ids, new_evaluation1s, new_evaluation2s, new_priority):
        # Validate input lengths match
        if not (len(new_data_list) == len(new_ids) == len(new_evaluation1s)):
            raise ValueError("Number of data items must match number of IDs")
        
        # Check for duplicate IDs
        duplicate_ids = set(new_ids) & set(self.ids)
        if duplicate_ids:
            raise ValueError(f"Duplicate IDs found: {duplicate_ids}")
        
        # Append new data and IDs
        self.data = np.append(self.data, new_data_list)
        self.ids = np.append(self.ids, new_ids)
        self.evaluation1s = np.append(self.evaluation1s, new_evaluation1s)
        self.evaluation2s = np.append(self.evaluation2s, new_evaluation2s)
        self.priority = np.append(self.priority, new_priority)
        
        # Update the index mapping
        self._update_id_to_index_mapping()
    
    def bulk_delete_items(self, ids_to_remove):
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
            self.data = np.delete(self.data, idx)
            self.ids = np.delete(self.ids, idx)
            self.evaluation1s = np.delete(self.evaluation1s, idx)
            self.evaluation2s = np.delete(self.evaluation2s, idx)
            self.priority = np.delete(self.priority, idx)
        
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