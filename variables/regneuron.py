import numpy as np
from neuromatch.variables import PSame, IndexMap
from neuromatch.find import find_candidates

def find_candidates(index_line: np.ndarray, ref_indexmaps: list[IndexMap]) -> np.ndarray:
    candidate_collection = [np.zeros_like(index_line)]
    for ref_map in ref_indexmaps:
        ref_map_temp = ref_map.content.astype(np.float64)
        ref_map_temp[np.where(ref_map_temp == 0)] = np.nan
        diff = (ref_map_temp.T - index_line.T).T
        n_same = np.nansum(np.where(diff == 0, 1, 0), axis=0)
        candidate_collection.append(ref_map[:, np.argmax(n_same)])

    candidate_collection = np.vstack(candidate_collection)
    uniq_candidates = np.unique(candidate_collection, axis=0)
    uniq_candidates = []
    for i in range(candidate_collection.shape[1]):
        uniq_candidates.append(list(np.unique(candidate_collection[:, i])))
        
    print(uniq_candidates)
    return candidate_collection

class RegisteredNeuron(object):
    def __init__(self, index_line: np.ndarray, p_same: PSame) -> None:
        self.content = index_line
        self.p_same = p_same
        self.n_session = self.content.shape[0]
        self._candidates = [[] for _ in range(self.n_session)]
    
    def find_candidates(self, ref_indexmaps: list[IndexMap]):
        """find_candidates: Get a candidate stack for further process.

        Parameters
        ----------
        ref_indexmaps : list[IndexMap]
            Additional index_maps to help re-match the index_line from all
            potential candidates.

        Returns
        -------
        np.ndarray
            _description_
        """
        candidates = [np.zeros_like(self.content)]
        for ref_map in ref_indexmaps:
            ref_map_temp = ref_map.content.astype(np.float64)
            ref_map_temp[np.where(ref_map_temp == 0)] = np.nan
            diff = (ref_map_temp.T - self.content.T).T
            n_same = np.nansum(np.where(diff == 0, 1, 0), axis=0)
            candidates.append(ref_map[:, np.argmax(n_same)])

        candidates = np.vstack(candidates)
        self.raw_candidates = candidates
        
        uniq_candidates = []
        for i in range(candidates.shape[1]):
            uniq_candidates.append(list(np.unique(candidates[:, i])))
            
        self.uniq_candidates = uniq_candidates
        
if __name__ == "__main__":
    from neuromatch.read import read_index_map
    
    index_map = read_index_map(r"E:\Data\Cross_maze\10227\Super Long-term Maze 1\Cell_reg\cellRegistered.mat")
    
    ref_indexmaps = [read_index_map(dir_name) for dir_name in [
        r"E:\Data\Cross_maze\10227\Super Long-term Maze 1\Ref "+str(i)+r"\cellRegistered.mat" for i in [1, 4, 7, 10, 15, 17, 20, 23, 26]
    ]]
    
    index_map.sort()
    idx = 32
    
    print(index_map[:, 32])
    find_candidates(index_map[:, 32], ref_indexmaps)