import numpy as np
import pandas as pd
import numpy as np
import os
import h5py
import scipy.io

def read_index_map(dir_name: str, open_type = 'h5py') -> np.ndarray:
    """ReadCellReg: Read the index_map from the MATLAB file saved by CellReg.
    CellReg: https://doi.org/10.1016/j.celrep.2017.10.013

    Parameters
    ----------
    dir_name : str
        The directory of the MATLAB file
    open_type : str, optional
        The method to open the MATLAB file, depending on the file signature
        Generally, 'h5py' and 'scipy' are used, by default 'h5py'
        'h5py' is needed when using v7.3 signature to save the file
        'scipy' is needed when 'h5py' does not work, relating to old-version MATLAB file.

    Returns
    -------
    index_map, numpy.ndarray with 2 dimensions (n_sessions, n_neurons)
        The index_map of neuron IDs.
        
    Raises
    ------
    FileNotFoundError
        If the directory does not exist
    """
    if os.path.exists(dir_name) == False:
        raise FileNotFoundError
    
    if open_type == 'h5py':
        with h5py.File(dir_name, 'r') as f:
            cell_registered_struct = f['cell_registered_struct']
            index_map = np.array(cell_registered_struct['cell_to_index_map'])
        return index_map.astype(np.int64)

    elif open_type == 'scipy':
        f = scipy.io.loadmat(dir_name)
        cell_registered_struct = f['cell_registered_struct']
        index_map = np.array(cell_registered_struct['cell_to_index_map'])
        return index_map.astype(np.int64)

def read_register_score(dir_name: str, open_type = 'h5py') -> np.ndarray:
    """read_register_score: Read the register scores from the MATLAB file saved by CellReg.
    CellReg: https://doi.org/10.1016/j.celrep.2017.10.013
    
    Concisely,
        Register score = 1 / ((N-1)*N') * sum(sum(delta(k, m)))
    where N is the total number of sessions, N' is the number of active sessions,
    delta(k, m) is the element (k, m) in the delta matrix whose member is 1 if cell-pair between
    session k and session m is reliable, and 0 otherwise. Only active-active and active-inactive 
    pairs are considered.

    Parameters
    ----------
    dir_name : str
        The directory of the MATLAB file
    open_type : str, optional
        The method to open the MATLAB file, depending on the file signature
        Generally, 'h5py' and 'scipy' are used, by default 'h5py'
        'h5py' is needed when using v7.3 signature to save the file
        'scipy' is needed when 'h5py' does not work, relating to old-version MATLAB file.

    Returns
    -------
    register_score, numpy.ndarray with 1 dimensions (n_neurons, )
        The register scores of registered neurons.
        
    Raises
    ------
    FileNotFoundError
        If the directory does not exist
    NotImplementedError
        If open type 'scipy' is used
    ValueError
        If open type is not 'h5py' or 'scipy'
    """
    if os.path.exists(dir_name) == False:
        raise FileNotFoundError
    
    if open_type == 'h5py':
        with h5py.File(dir_name, 'r') as f:
            cell_registered_struct = f['cell_registered_struct']
            register_score = np.array(cell_registered_struct['cell_scores'])
        return register_score[0]
    elif open_type == 'scipy':
        raise NotImplementedError("open type 'scipy' is not supported yet")
    else:
        raise ValueError("open type should be 'h5py' or 'scipy'")
    
def read_exclusivity_score(dir_name: str, open_type = 'h5py'):
    """read_exclusivity_score: Read the exclusive scores from the MATLAB file saved by CellReg.
    CellReg: https://doi.org/10.1016/j.celrep.2017.10.013
    
    Concisely,
        Exclusivity score = 1 / N' * sum(sum(delta(k, m)))
    where N is the total number of sessions, N' is the number of active sessions,
    delta(k, m) is the element (k, m) in the delta matrix whose member is 1 if cell-pair between
    session k and session m is reliable, and 0 otherwise. Only active-active pairs are considered.

    Parameters
    ----------
    dir_name : str
        The directory of the MATLAB file
    open_type : str, optional
        The method to open the MATLAB file, depending on the file signature
        Generally, 'h5py' and 'scipy' are used, by default 'h5py'
        'h5py' is needed when using v7.3 signature to save the file
        'scipy' is needed when 'h5py' does not work, relating to old-version MATLAB file.
            
    Returns
    -------
    exclusivity_score, numpy.ndarray with 1 dimensions (n_neurons, )
        The exclusivity scores of registered neurons.
        
    Raises
    ------
    FileNotFoundError
        If the directory does not exist
    NotImplementedError
        If open type 'scipy' is used
    ValueError
        If open type is not 'h5py' or 'scipy'
    """
    if os.path.exists(dir_name) == False:
        raise FileNotFoundError
    
    if open_type == 'h5py':
        with h5py.File(dir_name, 'r') as f:
            cell_registered_struct = f['cell_registered_struct']
            exclusivity_score = np.array(cell_registered_struct['exclusivity_scores'])
        return exclusivity_score[0]
    elif open_type == 'scipy':
        raise NotImplementedError("open type 'scipy' is not supported yet")
    else:
        raise ValueError("open type should be 'h5py' or 'scipy'")
    
def read_p_same(dir_name: str, open_type = 'h5py'):
    """read_p_same: Read the pair-wise P-same probability from the MATLAB file saved by CellReg.

    Parameters
    ----------
    dir_name : str
        The directory of the MATLAB file
    open_type : str, optional
        The method to open the MATLAB file, depending on the file signature
        Generally, 'h5py' and 'scipy' are used, by default 'h5py'
        'h5py' is needed when using v7.3 signature to save the file
        'scipy' is needed when 'h5py' does not work, relating to old-version MATLAB file.
            
    Returns
    -------
    p_same, numpy.ndarray with 3 dimensions (n_neurons, n_session, n_session)
        The pair-wise P-same of registered neurons.
        
    Raises
    ------
    FileNotFoundError
        If the directory does not exist
    NotImplementedError
        If open type 'scipy' is used
    ValueError
        If open type is not 'h5py' or 'scipy'
    """
    if os.path.exists(dir_name) == False:
        raise FileNotFoundError
    
    if open_type == 'h5py':
        with h5py.File(dir_name, 'r') as f:
            cell_registered_struct = f['cell_registered_struct']

            for element in cell_registered_struct['p_same_registered_pairs']:
                # !!! The length of element is n_neuron, (It only iterates for 1 time, thereby it's not a real for loop albeit a similar 'form')
                p_same_single = [f[element[i]][:] for i in range(len(element))]
                p_same = np.stack(p_same_single, axis=0)
        return p_same
    elif open_type == 'scipy':
        raise NotImplementedError("open type 'scipy' is not supported yet")
    else:
        raise ValueError("open type should be 'h5py' or 'scipy'")


def read_all_to_all_indixes(dir_name: str, open_type = 'h5py') -> list[list[list[list]]]:
    """read_all_to_all_indixes: Read the all-to-all indice from the MATLAB file saved by CellReg.

    Parameters
    ----------
    dir_name : str
        The directory of the MATLAB file
    open_type : str, optional
        The method to open the MATLAB file, depending on the file signature
        Generally, 'h5py' and 'scipy' are used, by default 'h5py'
        'h5py' is needed when using v7.3 signature to save the file
        'scipy' is needed when 'h5py' does not work, relating to old-version MATLAB file.
            
    Returns
    -------
    all_to_all_indixes, very complex list hierarchy.
    
    The original data structure is MATLAB 1*n_sessions MATLAB Cell, and each cell contains a 
    n_neurons*n_sessions MATLAB Cell, the later contains a list.
        
    Raises
    ------
    FileNotFoundError
    """
    if os.path.exists(dir_name) == False:
        raise FileNotFoundError
    
    if open_type == 'h5py':
        with h5py.File(dir_name, 'r') as f:
            modeled_data_struct = f['modeled_data_struct']
            print(modeled_data_struct.keys())
            all_to_all_indexes = [f[index[0]][:] for index in modeled_data_struct['all_to_all_indexes']]
            

            for index in modeled_data_struct['all_to_all_indexes']:
                
                # Iteration for 26 loops to travel all of the 1*26 MATLAB Cells
                for index_single_day in f[index[0]]:
                    # Iteration for 26 loops, each index_single_day has a length of n_neurons (e.g., 590)
                    for index_single_day_single_neuron in index_single_day:
                        # Iteration for 26 loops, each index_single_day_single_neuron has a length of n_sessions (e.g., 2)
                        print(len(index_single_day_single_neuron))
                        break
                    break

                break

def read_footprint(dir_name: str, open_type = 'h5py', key_word: str = 'SFP') -> np.ndarray:
    """read_footprint: Read the spatial footprint generated by
    CNMF-E (https://doi.org/10.7554/eLife.28728)
    or suit2P (https://doi.org/10.1101/061507).

    Parameters
    ----------
    dir_name : str
        The directory of the footprint
    open_type : str, optional
        The method to open the MATLAB file, depending on the file signature
        Generally, 'h5py' and 'scipy' are used, by default 'h5py'
        'h5py' is needed when using v7.3 signature to save the file
        'scipy' is needed when 'h5py' does not work, relating to old-version MATLAB file.
    key_word : str, optional
        key work to reach the footprint matrix, by default 'SFP'

    Returns
    -------
    np.ndarray
        The footprint matrix, three dimensions (n_neurons, n_pixels width, n_pixels height)

    Raises
    ------
    FileNotFoundError
        If the directory does not exist
    ValueError
        If the open_type is not 'h5py' or 'scipy'
    """
    if os.path.exists(dir_name) == False:
        raise FileNotFoundError
    
    if open_type == 'h5py':
        with h5py.File(dir_name, 'r') as f:
            sfp = np.array(f['SFP'])
        return sfp
    elif open_type == 'scipy':
        f = scipy.io.loadmat(dir_name)
        sfp = np.array(f['SFP'])
        return sfp
    else:
        raise ValueError("open type should be 'h5py' or 'scipy'")
    
    
if __name__ == '__main__':
    dir_name = r"E:\Data\Cross_maze\10227\Super Long-term Maze 1\Cell_reg\modeled_data_struct.mat"
    read_all_to_all_indixes(dir_name=dir_name)
    
    dir_name = r"E:\Data\Cross_maze\10227\Super Long-term Maze 1\Cell_reg\cellRegistered.mat"
    
    # Test
    index_map = read_index_map(dir_name)
    print("index_map:", type(index_map), index_map.shape)
    
    register_score = read_register_score(dir_name)
    print("register_score:", type(register_score), register_score.shape)
    
    exclusivity_score = read_exclusivity_score(dir_name)
    print("exclusivity_score:", type(exclusivity_score), exclusivity_score.shape)
    
    p_same = read_p_same(dir_name)
    print("p_same:", type(p_same), p_same.shape)
    