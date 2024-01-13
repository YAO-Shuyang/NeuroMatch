import numpy as np

class IndexMap(object):
    """IndexMap class stores the cell_to_index_map saved by CellReg,
    and provide related methods for sorting, etc."""
    def __init__(self, content: np.ndarray) -> None:
        self.n_neuron = content.shape[1]
        self._n_session = content.shape[0]
        self.content = content
        self.count_nonzero()
    
    @property
    def n_session(self):
        return self._n_session
    
    def count_nonzero(self) -> np.ndarray:
        """count_nonzero: Count the number of non-zero elements in each 
        registered neuron group.

        Returns
        -------
        np.ndarray, 1D with length of n_neuron
            the number of non-zero elements in each registered neuron group.
        """
        self.n_aligned = np.count_nonzero(self.content, axis=0)
        return self.n_aligned
    
    def __setitem__(self, index: int | tuple, value: int) -> int | np.ndarray:
        self.content[index] = value
    
    def __getitem__(self, index: int | tuple) -> int | np.ndarray:
        return self.content[index]
    
    def sort(self):
        # Count the number of non-zero elements in each row
        nonzero_count = np.count_nonzero(self.content, axis=0)
        # Sort the rows based on the count of non-zero elements
        sorted_indexes = np.argsort(nonzero_count)[::-1]
        self.content = self.content[:, sorted_indexes]
        self.count_nonzero()
        return self.content
    
    @staticmethod
    def sort_index_map(index_map: np.ndarray)-> np.ndarray:
        """sort_index_map: Sort the index map based on the number of non-zero elements in each column.

        Parameters
        ----------
        index_map : np.ndarray
            The index map to be sorted.

        Returns
        -------
        np.ndarray
            The sorted index map.
        """
        # Count the number of non-zero elements in each row
        nonzero_count = np.count_nonzero(index_map, axis=0)
        
        # Sort the rows based on the count of non-zero elements
        sorted_indices = np.argsort(nonzero_count)[::-1]
        sorted_index_map = index_map[:, sorted_indices]
        return sorted_index_map
    
    def shape(self):
        return self.content.shape
    
    def __len__(self):
        return len(self.content)
    
    def __str__(self):
        return str(self.content)
    
    def __repr__(self):
        return repr(self.content)
    
    def __iter__(self):
        return iter(self.content)
    
    def add_col(self, col: np.ndarray):
        self.content = np.hstack((self.content, col))
        self.update()
        return self.content
        
    def remove_col(self, index: int):
        self.content = np.delete(self.content, index, axis=1)
        self.update()
        return self.content
    
    def update(self):
        self.n_neuron = self.content.shape[1]
        self.n_session = self.content.shape[0]
        self.count_nonzero()
        return self.content