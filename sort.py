import numpy as np


def sort(index_map: np.ndarray):
    # Count the number of non-zero elements in each row
    nonzero_count = np.count_nonzero(index_map, axis=0)
    
    # Sort the rows based on the count of non-zero elements
    sorted_indices = np.argsort(nonzero_count)
    sorted_matrix = index_map[:, sorted_indices]
    
    return sorted_matrix