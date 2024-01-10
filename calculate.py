# To define the criteria of neuron re-matching.
import numpy as np

def calc_exclusivity_score(index_line: np.ndarray, p_same: np.ndarray, p_thre: float = 0.5) -> float:
    """calc_exclusivity_score: it is a function to calculate the exclusivity score.
    Refer to CellReg: https://doi.org/10.1016/j.celrep.2017.10.013
    and Github repository: https://github.com/zivlab/CellReg/blob/master/CellReg/compute_scores.m
    
    Briefly,
        Exclusivity score = 1 / N * sum(I(P_same >= p_thre)),
    where N is the total number of active sessions, and I is the indicator function, whose value
    is 1 if P_same >= p_thre, and 0 otherwise. Sum is taken over all active sessions and the result 
    is averaged.

    Parameters
    ----------
    index_line : np.ndarray
        A registered neuron index line.
    p_same : np.ndarray, shape (n_session, n_session). 
        The pair-wise P-same of registered neurons. If active-inactive or inactive-inactive pairs, the value is nan.
    p_thre : float, optional
        The threshold of P-same, by default 0.5
        
    Returns
    -------
    float
        The exclusivity score.
    """

    # Only neurons being detected in more than 2 sessions are considered.
    if len(np.where(index_line != 0)[0]) <= 1:
        return np.nan
    
    tri_p_same = np.triu(p_same, k=1)
    return np.where(tri_p_same >= p_thre)[0].shape[0] / np.where(tri_p_same>0)[0].shape[0]

