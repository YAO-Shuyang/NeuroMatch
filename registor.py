import numpy as np
from dataclasses import dataclass
from neuromatch.read import read_index_map, read_p_same, read_all_to_all_indexes, read_all_to_all_psame
from neuromatch.variables import AllToAll, IndexMap

class CellRegData(object):
    """CellRegData class reads and stores data in CellReg results, which are 
    required for re-matching."""
    def __init__(
        self, 
        registered_dir: str = r".\cellRegistered.mat", 
        model_dir: str = r".\modeled_data_struct.mat",
        p_model: str = 'spatial_correlation_model'
    ) -> None:
        ""
        self._registered_dir = registered_dir
        self._model_dir = model_dir
        self._read_data()
    
    @property
    def registered_dir(self):
        return self._registered_dir
    
    @property
    def model_dir(self):
        return self._model_dir
        
    def _read_data(self)-> None:
        """_read_data: read the cellRegistered.mat and modeled_data_struct.mat
        """
        self.index_map = read_index_map(self._registered_dir)
        self.n_sessions = self.index_map.shape[0]
        self.n_neurons = self.index_map.shape[1]
        
        self.p_same = read_p_same(self._registered_dir)
        self.ata_indexes = AllToAll(read_all_to_all_indexes(self._model_dir))
        self.ata_p_same = AllToAll(read_all_to_all_psame(self._model_dir))
    

class DataRepository(object):
    """DataRepository class stores basic information that is required for
    neuron re-match.
    
    Parameters
    ----------
    main_directory: str
        You have to select a CellReg file as the majoy file and all the modifications
        would be applied to this file.
    ref_directory: list[str]
        You have to provide additional CellReg files, basically those chosen distinct
        reference sessions. Our re-match algorithms will use these files as auxiliary
        information.
        
    Attributes
    ----------
    
    """
    main_directory: str
    ref_directory: list[str]
    ref_sessions: list[int]
    
    index_map: np.ndarray | None = None
    p_same: np.ndarray | None = None
    p_thre: float = 0.05
    register_score: np.ndarray | None = None
    
    def init(self):
        """init the data repository, including reading the data and othre pre-processing.
        """
        self.index_map = read_index_map(self.main_directory)
        self.n_sessions = self.index_map.shape[0]
        self.n_neurons = self.index_map.shape[1]
        self.p_same = read_p_same(self.main_directory)
        self.sorted_index_map = self.sort()
        
        self.index_map_all = []
        for i in range(len(self.ref_directory)):
            self.index_map_all.append(read_index_map(self.ref_directory))
        
        self._iter_i = 0
        
    def sort(self, index_map: np.ndarray | None = None) -> np.ndarray:
        """sort: the data repository provides method to sort the index_map,
        either the innately setted index_map or the inputted index_map

        Parameters
        ----------
        index_map : np.ndarray | None, optional
            The index map to be sorted, by default None
            If it is none, sort the innately setted index_map

        Returns
        -------
        np.ndarray
            The sorted index map
        """

        
    def calc_register_score(self) -> np.ndarray:
        """calc_register_score: calculate the register score of each neuron
        """
        self.register_score = np.zeros(self.n_neurons)
        for i in range(self.n_neurons):
            self.register_score[i] = calc_register_score(self.index_map[:, i], 
                                                         self.p_same[i, :, :], self.p_thre)
        return self.register_score
        
    def update(self):
        if self._iter_i >= self.n_neurons:
            return False

        self._iter_i += 1

@dataclass
class Registor(object):
    index_line: np.ndarray
    register_score: float
    n_sessions: int
    p_same: np.ndarray
    p_thre: float = 0.05
    
    uncertain_pos: np.ndarray | None = None
    
    def find_uncertain_neurons(self) -> np.ndarray:
        """find_uncertain_neurons: find the neurons who's not currently present in the index_line.
        """
        self.uncertain_pos = np.where(self.index_line == 0)[0]
        return self.uncertain_pos
    
    def find_all_candidates(self, index_map_all: list[np.ndarray], p_same):
        
    
    def calc_score(self, index_line: np.ndarray, p_same: np.ndarray) -> float:
        register_score = calc_register_score(self.index_line, self.p_same, self.p_thre)
        return register_score
    
    
    