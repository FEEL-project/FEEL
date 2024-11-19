import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class EventDataset(Dataset):
    def __init__(self, data=None, ids=None, evaluations=None):
        # Initialize with empty lists if no data provided
        self.data = np.array(data) if data is not None else np.empty((0,), dtype=object)
        self.ids = np.array(ids) if ids is not None else np.empty((0,), dtype=object)
        self.evaluations = np.array(evaluations) if evaluations is not None else np.empty((0,), dtype=object)
        
        # Validate input consistency
        if not (len(self.data) == len(self.ids) == len(self.evaluations)):
            raise ValueError("Data, IDs, and evaluations must have the same length")
        
        # Initialize id to index mapping
        self._update_id_to_index_mapping()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {
            'id': self.ids[idx], 
            'data': torch.from_numpy(self.data[idx]).float(),
            'evaluation': torch.from_numpy(self.evaluations[idx]).float()
        }

    def _update_id_to_index_mapping(self):
        # Rebuild the id to index mapping (for thread safety)
        self._id_to_index.clear()
        self._id_to_index = {id_val: idx for idx, id_val in enumerate(self.ids)}

    def get_by_id(self, id_value):
        # {'id': self.ids[idx], 
        #  'data': torch.from_numpy(self.data[idx]).float(), 
        #  'evaluation': torch.from_numpy(self.evaluations[idx]).float()
        # }
        index = self._id_to_index.get(id_value)
        return self[index] if index is not None else None

    def add_item(self, new_data, new_id, new_evaluation):
        # Ensure new_data is a numpy array
        if not isinstance(new_data, np.ndarray):
            new_data = np.array(new_data)
        
        # Check for duplicate ID
        if new_id in self._id_to_index:
            raise ValueError(f"ID {new_id} already exists in the dataset")
        
        # Append new data and ID
        self.data = np.append(self.data, new_data)
        self.ids = np.append(self.ids, new_id)
        self.evaluations = np.append(self.evaluations, new_evaluation)
        
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
        self.evaluations = np.delete(self.evaluations, index)
        
        # Rebuild the index mapping
        self._update_id_to_index_mapping()

    def bulk_add_items(self, new_data_list, new_ids, new_evaluations):
        # Validate input lengths match
        if not (len(new_data_list) == len(new_ids) == len(new_evaluations)):
            raise ValueError("Number of data items must match number of IDs")
        
        # Check for duplicate IDs
        duplicate_ids = set(new_ids) & set(self.ids)
        if duplicate_ids:
            raise ValueError(f"Duplicate IDs found: {duplicate_ids}")
        
        # Append new data and IDs
        self.data = np.append(self.data, new_data_list)
        self.ids = np.append(self.ids, new_ids)
        self.evaluations = np.append(self.evaluations, new_evaluations)
        
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
            self.evaluations = np.delete(self.evaluations, idx)
        
        # Update the index mapping
        self._update_id_to_index_mapping()