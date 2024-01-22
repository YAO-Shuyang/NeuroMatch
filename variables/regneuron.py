import numpy as np
from numba import jit
import copy as cp
from neuromatch.variables import PSame, IndexMap, AllToAll, AllToAllList
from neuromatch.find import find_candidates
import warnings

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
        
    print(uniq_candidates, sum(uniq_candidates,[]))
    return candidate_collection

#@jit(nopython=True)
def get_psame(index_line: np.ndarray, all_psame: np.ndarray, candidates: np.ndarray):
    """get_psame: get psame matrix for every index_line.

    Parameters
    ----------
    index_line : np.ndarray, with a shape of (n_session, )
        The index line currently being considered, whose corresponding psame will be computed.
    all_psame : np.ndarray
        The psame matrix of all candidates, with a shape of (n_candidates, n_candidates).
    candidates : np.ndarray
        The vectorized candidate matrix, with a shape of (n_candidates, 2)

    Returns
    -------
    p_same: np.ndarray
        The psame matrix of the index line, with a shape of (n_session, n_session).
    """
    index = np.zeros_like(index_line, np.int64)
    for i in range(index_line.shape[0]):
        index[i] = np.where((candidates[:, 0] == index_line[i])&(candidates[:, 1] == i))[0][0]
    
    psame = all_psame[index, :][:, index]
    return psame

@jit(nopython=True)
def calc_pmean_score(index_line: np.ndarray, psame: np.ndarray, frac: float = 0.8):
    """calc_pmean_score: compute a mean value to estimate the results of re-matching.

    Parameters
    ----------
    index_line : np.ndarray
    
    psame : np.ndarray
        The psame matrix of the index line, with a shape of (n_session, n_session).
    frac : float, optional
        The structural limitation which encourage matching for more neurons, by default 0.8

    Returns
    -------
    float
        The pmean score of a given index line
    """
    _len = len(psame)
    a = np.zeros(_len**2, dtype=np.float64)
    for i in range(_len):
        a[i*_len:(i+1)*_len] = psame[i, :]

    return np.mean(a[np.where(np.isnan(a) == False)[0]])*frac + np.where(index_line!=0)[0].shape[0]/index_line.shape[0]*(1-frac)

#@jit(nopython=True)
def iteration(
    index_line: np.ndarray, 
    all_psame: np.ndarray, 
    candidates: np.ndarray, 
    init_value: float,
    frac: float = 0.8
) -> np.ndarray:
    """iteration: do one iteration of re-matching

    Parameters
    ----------
    index_line : np.ndarray
        The index line to be updated
    all_psame : np.ndarray
        vectorized psame matrix for between each two candidates.
        With a shape of (n_candidates, n_candidates)
    candidates : np.ndarray
        the vecteroized candidates matrix, with a shape of (n_candidates, 2)
    init_value : float
        The initial value of pmean score    

    Returns
    -------
    np.ndarray
        The updated index line
    """
    max_value = init_value
    for i in range(index_line.shape[0]):
        idx = np.where(candidates[:, 1] == i)[0]
        for j in idx:
            if index_line[i] != candidates[j, 0]:
                index_i = index_line[i] # Save the current value and feed back if this iteration do not find a better value
                index_line[i] = candidates[j, 0] # Update all the candidates at the index on session i+1
                _psame = get_psame(index_line=index_line, all_psame=all_psame, candidates=candidates) # Compute the psame for the updated index map
                _score = calc_pmean_score(index_line=index_line, psame=_psame, frac=frac) # Compute the pmean score
                if _score > max_value:
                    # If the pmean score is better, update the index line
                    max_value = _score
                else:
                    # If the pmean score is not better, restore the index line
                    # and feed back the current value
                    index_line[i] = index_i
        
    return index_line, max_value

#@jit(nopython=True)
def optimization(
    index_line: np.ndarray, 
    all_psame: np.ndarray, 
    candidates: np.ndarray, 
    init_value: float,
    max_iter: int = 20,
    frac: float = 0.8
) -> np.ndarray:
    """optimization: Optimize the registered neuron with the found candidates

    Parameters
    ----------
    index_line : np.ndarray
        The initial index_line to be optimized
    all_psame : np.ndarray
        vectorized psame matrix for between each two candidates.
        With a shape of (n_candidates, n_candidates)
    candidates : np.ndarray
        the vecteroized candidates matrix, with a shape of (n_candidates, 2)
    init_value : float
        The initial value of pmean score
    max_iter : int, optional
        The optimization is repeated until the iterations reach max_iter, by default 20

    Returns
    -------
    np.ndarray
        The optimized index line
    """
    for i in range(max_iter):
        _index_line, init_value = iteration(
            index_line=index_line, 
            all_psame=all_psame, 
            candidates=candidates, 
            init_value=init_value, 
            frac=frac
        )
        if np.sum(np.abs(_index_line - index_line)) == 0:
            # If the results have converged, stop the optimization
            break
        index_line = _index_line
    return index_line
    

class RegisteredNeuron(object):
    def __init__(
        self, 
        index_line: np.ndarray, 
        ref_indexmaps: list[IndexMap], 
        ata_p_sames: AllToAllList, 
        ata_indexmaps: AllToAllList,
        p_thre: float = 0.5
    ) -> None:
        self.content = cp.deepcopy(index_line)
        self.ori_content = cp.deepcopy(index_line)
        self.n_session = self.content.shape[0]
        self.p_thre = p_thre
        print(" 1. Done initialization.")
        self._find_candidates(ref_indexmaps=ref_indexmaps)
        print(" 2. All candidates were found.")
        self._vectorize_candidates(self._raw_candidates)
        print(" 3. Done vectorization of candidates.")
        self._vectorize_ata_psame_matrix(ata_p_sames, ata_indexmaps)
        print(" 4. Done vectorization of psame matrix")
        self._init_psame()
        print(" 5. Done computing of initialized psame.")
        self._init_score()
        print(" 6. Done computing of initialized score.")
        
    def _vectorize_candidates(self, candidate: list[list]) -> np.ndarray:
        """_vectorize_candidates: vectorize candidates object into a 2D array
        to speed up the computation.

        Parameters
        ----------
        candidate : list[list]
            The object to be vectorized, with length of the outlier list is n_session,
            while the 

        Returns
        -------
        np.ndarray, 2D array, with shape (ID, 2)
        """
        vec_candidate = []
        for i in range(len(candidate)):
            for j in candidate[i]:
                vec_candidate.append([j, i])
        
        return np.array(vec_candidate, np.int64)
    
    def _vectorize_ata_psame_matrix(self, ata_p_sames: AllToAllList, ata_indexmaps: AllToAllList):
        """_vectorize_ata_psame_matrix: vectorize the neuron candidate-wise P-same
        for speed up.

        Parameters
        ----------
        ata_p_sames : AllToAllList
            All-to-all list of the P-same matrix. It contains the main all-to-all pvalue and 
        """
        n = self._uniq_candidates.shape[0]
        all_psame = np.full((n, n), np.nan)
        for i in range(n-1):
            for j in range(i+1, n):
                if self._uniq_candidates[i, 1] == self._uniq_candidates[j, 1]:
                    continue
                
                if self._uniq_candidates[i, 0] == 0 or self._uniq_candidates[j, 0] == 0:
                    continue
                
                NA, NB = self._uniq_candidates[i, 0], self._uniq_candidates[j, 0] # Neuron A, B
                SA, SB = self._uniq_candidates[i, 1], self._uniq_candidates[j, 1] # Session A, B
                
                for k in range(len(ata_p_sames)):
                    if NB in ata_indexmaps[k, SA, SB, NA-1]:
                        if np.isnan(all_psame[SA, SB]):
                            all_psame[i,j] = ata_p_sames[k, SA, SB, NA-1, np.where(np.array(ata_indexmaps[k,SA, SB, NA-1])==NB)[0][0]]
                        else:        
                            all_psame[i, j] = max(all_psame[SA, SB], ata_p_sames[k, SA, SB, NA-1, np.where(np.array(ata_indexmaps[k, SA, SB, NA-1])==NB)[0][0]])
                    else:
                        all_psame[i, j] = 0
            
        self.all_psame = all_psame
        
    def _init_psame(self):
        self.psame = get_psame(self.content, self.all_psame, self._uniq_candidates)
    
    def _init_score(self):
        self.value = calc_pmean_score(index_line=self.content, psame=self.psame, frac=self.p_thre)
        
    def _find_candidates(self, ref_indexmaps: list[IndexMap]):
        """find_candidates: Get a candidate stack for further process.

        Parameters
        ----------
        ref_indexmaps : list[IndexMap]
            Additional index_maps to help re-match the index_line from all
            potential candidates.
        """
        candidates = [cp.deepcopy(self.content)]
        for ref_map in ref_indexmaps:
            ref_map_temp = ref_map.content.astype(np.float64)
            ref_map_temp[np.where(ref_map_temp == 0)] = np.nan
            diff = (ref_map_temp.T - self.content.T).T
            n_same = np.nansum(np.where(diff == 0, 1, 0), axis=0)
            candidates.append(ref_map[:, np.argmax(n_same)])

        candidates = np.vstack(candidates)
        self._raw_candidates = candidates
        
        uniq_candidates = []
        for i in range(candidates.shape[1]):
            uniq_candidates.append(list(np.unique(candidates[:, i])))
            
        self._uniq_candidates = self._vectorize_candidates(uniq_candidates)

    def _assess_results(self):
        pass
    
    def optimize(self, max_iter: int = 20) -> np.ndarray:
        """optimize: Optimize the registered neuron with the found candidates

        Parameters
        ----------
        index_map : IndexMap
            The main index_map. Whenever we integrate a new candidate into the
            original index_map, we need to delete it from its previous position.
            Whenever we replace an old element A with candidate B, we have to find
            a new position to bear A (most likely to list it, together with other 
            replaced elements, in a new column of index_map attached at the end).
        max_iter : int, optional, by default 20
            The optimization is repeated until the iterations reach max_iter, or 
            the result converges. 
        """
        self.opt_content = optimization(
            index_line=cp.deepcopy(self.content),
            all_psame=self.all_psame,
            candidates=self._uniq_candidates,
            init_value=cp.deepcopy(self.value),
            max_iter=max_iter,
            frac=self.p_thre
        )
        return self.opt_content
        
        
if __name__ == "__main__":
    """
    from neuromatch.read import read_index_map, read_all_to_all_psame, read_all_to_all_indexes
    from neuromatch.variables.regneuron import RegisteredNeuron
    import pickle
    
    model_dir = r"E:\Data\Cross_maze\10227\Super Long-term Maze 1\Cell_reg\modeled_data_struct.mat"
    index_map = read_index_map(r"E:\Data\Cross_maze\10227\Super Long-term Maze 1\Cell_reg\cellRegistered.mat")
    
    print(index_map[:, 2])
    
    ref_dirs = [
        r"E:\Data\Cross_maze\10227\Super Long-term Maze 1\Ref "+str(i)+r"\cellRegistered.mat" for i in [1, 4, 7, 10, 15, 17, 20, 23, 26]
    ]
    ref_model_dir = [
        r"E:\Data\Cross_maze\10227\Super Long-term Maze 1\Ref "+str(i)+r"\modeled_data_struct.mat" for i in [1, 4, 7, 10, 15, 17, 20, 23, 26]
    ]

    ref_indexmaps = [read_index_map(dir_name) for dir_name in ref_dirs]
    ata_p_sames = [read_all_to_all_psame(dir_name) for dir_name in [model_dir]+ref_model_dir]
    ata_indexmaps = [read_all_to_all_indexes(dir_name) for dir_name in [model_dir]+ref_model_dir]
    with open(r"E:\Anaconda\envs\maze\Lib\site-packages\neuromatch\temp.pkl", 'wb') as f:
    pickle.dump([index_map, ref_indexmaps, ata_p_sames, ata_indexmaps], f)
    """
    from neuromatch.variables.regneuron import RegisteredNeuron
    from neuromatch.variables import AllToAllList
    import pickle
    with open(r"E:\Anaconda\envs\maze\Lib\site-packages\neuromatch\temp.pkl", 'rb') as f:
        index_map, ref_indexmaps, ata_p_sames, ata_indexmaps = pickle.load(f)

    reg_neuron = RegisteredNeuron(index_line=index_map[:, 2], ref_indexmaps=ref_indexmaps, ata_p_sames=AllToAllList(ata_p_sames), ata_indexmaps=AllToAllList(ata_indexmaps))
    print(reg_neuron.ori_content)
    reg_neuron.optimize()
    print(reg_neuron.opt_content)
    