import numpy as np

class AllToAll(object):
    """AllToAll class serves to store all_to_all index/p_same, providing
    methods to use its elements easily and logically."""
    def __init__(self, all_to_all: list[list[list[list[int | float]]]]) -> None:
        self._content = all_to_all
        
    @property
    def content(self) -> list[list[list[list[int | float]]]]:
        return self._content
    
    def __getitem__(self, index: tuple[int, int, int]) -> list[int]:
        neuron_session, target_session, neuron_id = index
        return self._content[neuron_session][target_session][neuron_id]