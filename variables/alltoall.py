from typing import Any
import numpy as np

class AllToAll(object):
    """AllToAll class serves to store all_to_all index/p_same, providing
    methods to use its elements easily and logically."""
    def __init__(self, all_to_all: list[list[list[list[int | float]]]]) -> None:
        self._content = all_to_all
        
    @property
    def content(self) -> list[list[list[list[int | float]]]]:
        return self._content
    
    def __getitem__(
        self, 
        index: int | tuple[int, int] | tuple[int, int, int] | tuple[int, int, int, int]
    ) -> list[list[list[int | float]]] | list[list[int | float]] | list[int | float] | int | float:
        """n_session * n_session * n_neuron_per_session * n_matched_neuron_per_neuron_per_session"""
        if isinstance(index, int):
            return self._content[index]
        elif isinstance(index, tuple) and len(index) == 2:
            return self._content[index[0]][index[1]]
        elif isinstance(index, tuple) and len(index) == 3:
            return self._content[index[0]][index[1]][index[2]]
        elif isinstance(index, tuple) and len(index) == 4:
            return self._content[index[0]][index[1]][index[2]][index[3]]
        else:
            raise TypeError(f"index must be int, tuple[int, int], tuple[int, int, int] or tuple[int, int, int, int], but got {type(index)}")
    
    def __setitem__(
        self, 
        index: int | tuple[int, int] | tuple[int, int, int] | tuple[int, int, int, int],
        value: list[list[list[int | float]]] | list[list[int | float]] | list[int | float] | int | float
    ) -> None:
        if isinstance(index, int):
            self._content[index] = value
        elif isinstance(index, tuple) and len(index) == 2:
            self._content[index[0]][index[1]] = value
        elif isinstance(index, tuple) and len(index) == 3:
            self._content[index[0]][index[1]][index[2]] = value
        elif isinstance(index, tuple) and len(index) == 4:
            self._content[index[0]][index[1]][index[2]][index[3]] = value
        else:
            raise TypeError(f"index must be int, tuple[int, int], tuple[int, int, int] or tuple[int, int, int, int], but got {type(index)}")
        
    def __len__(self) -> int:
        return len(self._content)
    
    def __str__(self) -> str:
        return str(self._content)
    
    def __repr__(self) -> str:
        return repr(self._content)
    
    def __iter__(self):
        return iter(self._content)
    
    

class AllToAllList(object):
    """AllToAllList class serves to store all all_to_all lists, providing
    methods to conviently process it."""
    def __init__(self, all_to_alls: list[AllToAll]) -> None:
        self._content = all_to_alls
    
    @property
    def content(self) -> list[AllToAll]:
        return self._content
    
    def __getitem__(
        self, 
        index: int | tuple[int, int] | tuple[int, int, int] | tuple[int, int, int, int] | tuple[int, int, int, int, int]
    ) -> list[list[list[list[int | float]]]] | list[list[list[int | float]]] | list[list[int | float]] | list[int | float] | int | float:
        """n_ref * n_session * n_session * n_neuron_per_session * n_matched_neuron_per_neuron_per_session"""
        if isinstance(index, int):
            return self._content[index].content
        elif isinstance(index, tuple) and len(index) == 2:
            return self._content[index[0]][index[1]]
        elif isinstance(index, tuple) and len(index) == 3:
            return self._content[index[0]][index[1], index[2]]
        elif isinstance(index, tuple) and len(index) == 4:
            return self._content[index[0]][index[1], index[2], index[3]]
        elif isinstance(index, tuple) and len(index) == 5:
            return self._content[index[0]][index[1], index[2], index[3], index[4]]
        else:
            raise TypeError(f"index must be int, tuple[int, int], tuple[int, int, int] or tuple[int, int, int, int] or tuple[int, int, int, int, int], but got {type(index)}")
    
    def __setitem__(
        self, 
        index: int | tuple[int, int] | tuple[int, int, int] | tuple[int, int, int, int] | tuple[int, int, int, int, int],
        value: list[list[list[list[int | float]]]] | list[list[list[int | float]]] | list[list[int | float]] | list[int | float] | int | float
    ) -> None:
        if isinstance(index, int):
            self._content[index] = AllToAll(value)
        elif isinstance(index, tuple) and len(index) == 2:
            self._content[index[0]][index[1]] = value
        elif isinstance(index, tuple) and len(index) == 3:
            self._content[index[0]][index[1], index[2]] = value
        elif isinstance(index, tuple) and len(index) == 4:
            self._content[index[0]][index[1], index[2], index[3]] = value
        elif isinstance(index, tuple) and len(index) == 5:
            self._content[index[0]][index[1], index[2], index[3], index[4]] = value
        else:
            raise TypeError(f"index must be int, tuple[int, int], tuple[int, int, int] or tuple[int, int, int, int] or tuple[int, int, int, int, int], but got {type(index)}")
        
    def __len__(self) -> int:
        return len(self._content)
    
    def __str__(self) -> str:
        return str(self._content)
    
    def __repr__(self) -> str:
        return repr(self._content)
    
    def __iter__(self):
        return iter(self._content)
    
    