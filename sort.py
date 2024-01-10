import numpy as np

def sort(index_map: np.ndarray):
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
    sorted_index_map = sort(index_map)
    print(sorted_index_map)