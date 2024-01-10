import numpy as np

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


if __name__ == '__main__':
    from neuromatch.read import read_index_map
    index_map = read_index_map(r"E:\Data\Cross_maze\10227\Super Long-term Maze 1\Cell_reg\cellRegistered.mat")
    print(index_map)
    sorted_index_map = sort_index_map(index_map)
    print(sorted_index_map)