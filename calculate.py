# To define the criteria of neuron re-matching.
import numpy as np
from numba import jit

@jit(nopython=True)
def calc_register_score(psame: np.ndarray, p_thre: float = 0.05) -> float:
    """calc_register_score: it is a function to calculate the register score.
    Refer to CellReg: https://doi.org/10.1016/j.celrep.2017.10.013
    and Github repository: https://github.com/zivlab/CellReg/blob/master/CellReg/compute_scores.m
    
    Briefly,

        register score = 1 / N * sum(I(psame >= p_thre)),

    where N is the total number of active sessions, and I is the indicator function, whose value
    is 1 if psame >= p_thre, and 0 otherwise. Sum is taken over all active-active cell 
    pairs and the result is averaged.
    
    Note
    ----
    This score will not be the same as the results computed by CellReg, cause the formular is not
    exactly the same! Ours only uses the psame of active-active cell pairs as the data of active-inactive
    cell pairs is not available directly from CellReg files. It more like the exclusivity score
    computed in CellReg but ours uses all the a-a pairs instead of exclusive a-a pairs only.
    
    Our register score has a similar distribution as the CellReg's register score when the p_thre
    is small.
    
    Example
    -------
    >>> index_line = np.array([1, 2, 3, 4, 5, 6])
    >>> psame = np.array([[0.87, 0.94, 0.89, 0.59, 0.82, np.nan],
        A = [[np.nan    0.87    0.97    np.nan  0.35    np.nan  ],
             [0.87      np.nan  0.94    0.97    0.47    np.nan  ],
             [0.97      0.94    np.nan  0.97    np.nan  0.63    ],
             [np.nan    0.97    0.97    np.nan  0.59    np.nan  ],
             [0.35      0.47    np.nan  0.59    np.nan  0.82    ],
             [np.nan    np.nan  0.63    np.nan  0.82    np.nan  ]]
             
        The upper triangle of A will be extracted.
    >>>    triu(A) = 
            [[          0.87    0.97    np.nan  0.35    np.nan  ],
             [                  0.94    0.97    0.47    np.nan  ],
             [                          0.89    np.nan  0.63    ],
             [                                  0.59    np.nan  ],
             [                                          0.82    ],
             [                                                  ]],
             
        (0 will take all the remaining position of numpy.ndarray during implementing)
        and the mean value of all the remaining non-nan values which are greater than p_thre
        will be computed as the register score.

    Parameters
    ----------
    psame : np.ndarray, shape (n_session, n_session). 
        The pair-wise P-same of registered neurons. If active-inactive or inactive-inactive pairs, the value is nan.
    p_thre : float, optional
        The threshold of P-same, by default 0.5
        
    Returns
    -------
    float
        The register score.
    """
    tri_psame = np.triu(psame, k=1)
    be_div = np.where(tri_psame >= p_thre)[0].shape[0]
    if be_div == 0:
        return np.nan
    else:
        return  np.where(tri_psame >= p_thre)[0].shape[0] / be_div

def calc_mean_psame(index_line: np.ndarray, psame: np.ndarray) -> float:
    """calc_mean_psame: it is a function to compute all the 

    Parameters
    ----------
    index_line : np.ndarray
        A registered neuron index line.
    psame : np.ndarray, shape (n_session, n_session). 
        The pair-wise P-same of registered neurons. If active-inactive or inactive-inactive pairs, the value is nan.

    Returns
    -------
    float
        The mean value of all the remaining non-nan values which are greater than p_thre
    """
    # Only neurons being detected in more than 2 sessions are considered.
    if len(np.where(index_line != 0)[0]) <= 1:
        return np.nan
    
    tri_psame = np.triu(psame, k=1)
    return np.nanmean(tri_psame[np.where(tri_psame>0)])

def calc_reliability(i: int, index_line: np.ndarray, psame: np.ndarray) -> float:
    """calc_reliability: it is a function to compute the reliability that a neuron
    to be registered with other neurons.

    Parameters
    ----------
    i : int
        The index of the neuron
    index_line : np.ndarray
        A registered neuron index line.
    psame : np.ndarray, shape (n_session, n_session). 
        The pair-wise P-same of registered neurons. If active-inactive or inactive-inactive pairs, the value is nan.

    Returns
    -------
    float
        The reliability of the neuron in a registered neuron group.
    """
    if index_line[i] == 0:
        return np.nan
    
    return np.nanmean(psame[i, :])

if __name__ == '__main__':
    from neuromatch.read import read_register_score, read_index_map, read_psame, read_exclusivity_score
    import pandas as pd
    import  matplotlib.pyplot as plt
    
    dir_name = r"E:\Data\Cross_maze\10227\Super Long-term Maze 1\Cell_reg\cellRegistered.mat"
    
    index_map = read_index_map(dir_name)
    register_score = read_register_score(dir_name)
    exclusivity_score = read_exclusivity_score(dir_name)
    psame = read_psame(dir_name=dir_name)
    
    register_test = np.zeros(index_map.shape[1])
    mean_psame = np.zeros(index_map.shape[1])
    
    for i in range(index_map.shape[1]):
        index_line = index_map[:, i]
        register_test[i] = calc_register_score(psame[i, :, :], p_thre=0.05)
        mean_psame[i] = calc_mean_psame(index_line, psame[i, :, :])
    
    res = {"real": exclusivity_score, "test": register_test, "register_score": register_score, "mean p": mean_psame}

    D = pd.DataFrame(res)
    D.to_excel(r"E:\Data\maze_learning\PlotFigures\STAT_CellReg\neuromatch_register_score.xlsx")