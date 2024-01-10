import numpy as np
from numba import jit

# No jit takes ~13.5 seconds for 10000 times
# with jit takes ~ 6.5~7.5 seconds for 10000 times
@jit(nopython=True)
def find_candidates(
    i: int,
    index_line: np.ndarray, 
    index_maps: list[np.ndarray],
    ref_sessions: list[int] | np.ndarray
) -> np.ndarray:
    """find_candidates: find all potential candidates at empty position i in index_line

    Parameters
    ----------
    i : int
        The index of the empty position
    index_line : np.ndarray, shape = (n, ), where n is the number of aligned sessions.
        Sliced from index_map generated by CellReg.
    index_maps : list[np.ndarray]
        A list of all index_map generated by CellReg. These index_maps should be generated
        by distinct reference sessions, and provide sufficient information for finding
        candidates.
        Each index_map are 2-dimensional, with shape (n_sessions, n_neurons)
    ref_sessions : list[int]
        A list of reference sessions.

    Returns
    -------
    np.ndarray with shape (n, 3), where n is the number of potential candidates.
        Contains information of all potential candidates. The first column is the 
        indice of candidate neurons, the second column is the indice of ref. neurons
        and the third column is the indice of reference sessions.
    """
    
    n_neuron = index_line.shape[0]
    n_ref = len(index_maps)
    
    candidates = np.zeros((n_neuron*n_ref, 3))*np.nan
    for j in range(index_line.shape[0]):
        if j != i and index_line[j] != 0:
            for k in range(len(index_maps)):
                # find line with  in index_map k
                idx = np.where(index_maps[k][j, :] == index_line[j])[0][0]
                if index_maps[k][i, idx] != 0:
                    candidates[j*n_ref + k, 0] = index_maps[k][i, idx]
                    candidates[j*n_ref + k, 1] = index_line[j]
                    candidates[j*n_ref + k, 2] = ref_sessions[k]
    
    return candidates[np.where(np.isnan(candidates[:, 0]) == False)]

def find_weak_pos(
    index_line: np.ndarray, 
    p_same: np.ndarray, 
    p_thre: float = 0.5
) -> np.ndarray:
    """find_weak_pos: find positions in index_line, which is taken by either uncertain 
    neuron candidates or empty position.

    Parameters
    ----------
    index_line : np.ndarray, shape = (n, ), where n is the number of aligned sessions.
        Sliced from index_map generated by CellReg.
    p_same : np.ndarray, shape (n_session, n_session). 
        The pair-wise P-same of registered neurons. If active-inactive or inactive-inactive 
        pairs, the value is nan.
    p_thre : float, optional
        The threshold of P-same, by default 0.5

    Returns
    -------
    np.ndarray with shape (n, ), where n is the number of weak position.
    """

if __name__ == '__main__':
    from neuromatch.read import read_index_map
    
    index_map = read_index_map(r"E:\Data\Cross_maze\10227\Super Long-term Maze 1\Cell_reg\cellRegistered.mat")
    
    index_maps = [read_index_map(dir_name) for dir_name in [
        r"E:\Data\Cross_maze\10227\Super Long-term Maze 1\Ref "+str(i)+r"\cellRegistered.mat" for i in [1, 4, 7, 10, 15, 17, 20, 23, 26]
    ]]
    
    ref_sessions = np.array([1, 4, 7, 10, 15, 17, 20, 23, 26])
    
    import time
    t1 = time.time()
    for i in range(100000):
        candidates = find_candidates(10, index_map[:, 26], index_maps, ref_sessions)
    print(time.time() - t1, candidates)
