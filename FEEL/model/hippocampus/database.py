import os
import pickle
import numpy as np
import faiss
from typing import List, Tuple, Optional
import threading

class VectorDatabase():
    def __init__(self, dimension: int, index_type: str = "Flat"):
        """
        Args:
            dimension: Vector dimension
            index_type: "IVF" or "Flat". IVF is faster but approximate, Flat is exact but slower
        """
        super(VectorDatabase, self).__init__()
        self.dimension = dimension
        self.id_map = {}  # maps internal index to user-provided id
        self.reverse_id_map = {}  # maps user-provided id to internal index
        self.next_index = 0
        self.lock = threading.Lock()
        
        if index_type == "IVF":
            # IVFFlat index for better performance
            nlist = max(4, int(np.sqrt(1000)))  # number of clusters
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            self.index.nprobe = 1  # number of clusters to visit during search
        else:
            # Flat index for exact search
            self.index = faiss.IndexFlatL2(dimension)
        
        # Initialize index
        self.is_trained = False
    
    def add(self, id: int, vector: np.ndarray) -> None:
        """Add a vector with associated id."""
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)
        
        if vector.shape != (self.dimension,):
            raise ValueError(f"Vector dimension must be {self.dimension}")
        
        with self.lock:
            # Train index if necessary
            if not self.is_trained:
                if isinstance(self.index, faiss.IndexIVFFlat):
                    self.index.train(vector.reshape(1, -1))
                self.is_trained = True
            
            # Add vector to index
            self.index.add(vector.reshape(1, -1))
            
            # Update id mappings
            self.id_map[self.next_index] = id
            self.reverse_id_map[id] = self.next_index
            self.next_index += 1
    
    def remove(self, id: int) -> bool:
        """Remove a vector by id. Returns True if successful."""
        with self.lock:
            if id not in self.reverse_id_map:
                return False
            
            # Get the internal index
            internal_idx = self.reverse_id_map[id]
            
            # Remove from index
            if isinstance(self.index, faiss.IndexIVFFlat):
                # For IVF index, we need to rebuild
                vectors = []
                ids = []
                for i in range(self.index.ntotal):
                    if i != internal_idx:
                        # Get vector at index i
                        vector = self.index.reconstruct(i)
                        vectors.append(vector)
                        ids.append(self.id_map[i])
                
                # Reset index
                self.index.reset()
                self.id_map.clear()
                self.reverse_id_map.clear()
                self.next_index = 0
                
                # Re-add vectors
                if vectors:
                    vectors = np.stack(vectors)
                    for vec, id in zip(vectors, ids):
                        self.add(id, vec)
            else:
                # For Flat index, we can remove directly
                self.index.remove_ids(np.array([internal_idx]))
                del self.id_map[internal_idx]
                del self.reverse_id_map[id]
            
            return True
    
    def search(self, query_vector: np.ndarray, k: int = 5, id: int=-1) -> List[Tuple[int, float]]:
        """
        Search for k nearest neighbors.
        Returns list of (id, distance) tuples.
        """
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)
        
        if query_vector.shape != (self.dimension,):
            raise ValueError(f"Query vector dimension must be {self.dimension}")
        
        # Search index
        distances, indices = self.index.search(query_vector.reshape(1, -1), k+1) # +1 to exclude the query vector itself
        
        # Convert to list of (id, distance) tuples
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:  # -1 indicates no result found
                # Skip the query vector itself
                if id != -1 and self.id_map[idx] == id:
                    continue
                results.append((self.id_map[idx], float(dist)))
        
        return results

    def __len__(self) -> int:
        """Return number of vectors in database."""
        return self.index.ntotal
    
    def save_to_file(self, file_path: str) -> None:
        """Save index to file."""
        with self.lock:
            base, _ = os.path.splitext(file_path)
            faiss_file = f"{base}_index.faiss"
            faiss.write_index(self.index, faiss_file)
            with open(file_path, 'wb') as f:
                # Save the rest of the object
                pickle.dump({
                    'dimension': self.dimension,
                    'id_map': self.id_map,
                    'reverse_id_map': self.reverse_id_map,
                    'next_index': self.next_index,
                    'is_trained': self.is_trained
                }, f)
                
    @staticmethod
    def load_from_file(file_path:str):
        """Load index from file."""
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
            db = VectorDatabase(obj['dimension'])
            db.id_map = obj['id_map']
            db.reverse_id_map = obj['reverse_id_map']
            db.next_index = obj['next_index']
            db.is_trained = obj['is_trained']
            base, _ = os.path.splitext(file_path)
            faiss_file = f"{base}_index.faiss"
            db.index = faiss.read_index(faiss_file)
            return db