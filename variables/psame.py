import numpy as np
from neuromatch.calculate import calc_register_score

class PSame(object):
    """PSame class serves to store p_same matrix, providing
    methods to conviently process it."""
    def __init__(self, p_same: np.ndarray | None = None, dim: int | None = None, p_thre: float = 0.5) -> None:
        if p_same.shape[0] != p_same.shape[1]:
            raise ValueError("p_same must be a square matrix.")
        
        if p_thre < 0 or p_thre > 1:
            raise ValueError("p_thre must be between 0 and 1.")
        
        if p_same is None:
            if dim is None:
                raise ValueError("dim is required if you wanted to initialize an empty matrix.")
            else:
                self.content = np.full((dim, dim), np.nan)
        else:
            self.content = p_same
            
        self._n_session = p_same.shape[0]
        self.p_thre = p_thre
        self.calc_value()
        
    @property
    def n_session(self):
        return self._n_session
    
    def __setitem__(self, index: int | tuple, value: int) -> float | np.ndarray:
        self.content[index] = value
        return self.content
    
    def __getitem__(self, index: tuple) -> float:
        return self.content[index]
    
    def __str__(self) -> str:
        return str(self.content)
    
    def __repr__(self) -> str:
        return repr(self.content)
    
    def __iter__(self):
        return iter(self.content)
    
    def __len__(self):
        return len(self.content)
    
    def shape(self):
        return self.content.shape
    
    def calc_value(self) -> float:
        self.value = calc_register_score(self.content, self.p_thre)
        return self.value
    
    def get_active_cells(self) -> np.ndarray:
        nonzero = np.count_nonzero(self.content, axis=0)
        idx = np.argmax(nonzero)
        self.active_idx = np.sort(np.append(np.where(self.content[: idx] > 0)[0], idx))
        return self.active_idx
    
    def update(self, p_thre: float | None = 0.5):
        if p_thre is not None:
            self.p_thre = p_thre
            
        self.calc_value()
        return self.content

class PSameList(object):
    """PSameList class serves to store p_same list, providing
    methods to conviently process it."""
    def __init__(self, p_same: np.ndarray, p_thre: float) -> None:
        self.content = []
        for i in range(p_same.shape[0]):
            self.content.append(PSame(p_same[i, :, :], p_thre=p_thre))
        
        self.p_thre = p_thre
        self.get_values()
            
    def __getitem__(self, index: int | tuple[int, int] | tuple[int, int, int]) -> PSame:
        if isinstance(index, int):
            return self.content[index]
        elif isinstance(index, tuple) and len(index) == 2:
            i, j = index
            return self.content[i][j]
        elif isinstance(index, tuple) and len(index) == 3:
            neuron_id, i, j = index
            return self.content[neuron_id][i, j]
        else:
            raise TypeError("index must be int or tuple[int, int, int].")
        
    @property
    def shape(self):
        if len(self.content) == 0:
            return (0, )
        else:
            return (len(self.content), self.content[0].shape[0], self.content[0].shape[1])
            
    def __len__(self):
        return len(self.content)
    
    def __str__(self) -> str:
        return str(self.content)
    
    def __repr__(self) -> str:
        return repr(self.content)
    
    def __iter__(self):
        return iter(self.content)
    
    def insert(self, index: int, value: PSame):
        if not isinstance(value, PSame):
            raise TypeError("value must be PSame object.")
        
        self.content.insert(index, value)
        return self.content
    
    def pop(self, index: int):
        return self.content.pop(index)
    
    def append(self, value: PSame):
        if not isinstance(value, PSame):
            raise TypeError("value must be PSame object.")
        
        self.content.append(value)
        return self.content
    
    def update(self, p_thre: float | None = 0.5):
        if p_thre is not None:
            self.p_thre = p_thre

        for i in range(len(self.content)):
            self.content[i].update(p_thre=self.p_thre)
            self.values[i] = self.content[i].value
        return self.content
    
    def get_values(self):
        self.values = np.zeros(len(self.content), dtype=np.float64)
        for i in range(len(self.content)):
            self.values[i] = self.content[i].value
        return self.values