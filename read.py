import numpy as np
import pandas as pd
import numpy as np
import os
import h5py
import scipy.io

from neuromatch.variables import IndexMap, AllToAll, PSame, PSameList

def read_index_map(dir_name: str, open_type = 'h5py') -> IndexMap:
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
        return IndexMap(index_map.astype(np.int64))

    elif open_type == 'scipy':
        f = scipy.io.loadmat(dir_name)
        cell_registered_struct = f['cell_registered_struct']
        index_map = np.array(cell_registered_struct['cell_to_index_map'])
        return IndexMap(index_map.astype(np.int64))
    
    else:
        raise ValueError("open_type should be 'h5py' or 'scipy'")

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
    
def read_exclusivity_score(dir_name: str, open_type = 'h5py') -> np.ndarray:
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
    
def read_psame(dir_name: str, open_type = 'h5py', p_thre: float = 0.5) -> PSameList:
    """read_psame: Read the pair-wise P-same probability from the MATLAB file saved by CellReg.

    Parameters
    ----------
    dir_name : str
        The directory of the MATLAB file
    open_type : str, optional
        The method to open the MATLAB file, depending on the file signature
        Generally, 'h5py' and 'scipy' are used, by default 'h5py'
        'h5py' is needed when using v7.3 signature to save the file
        'scipy' is needed when 'h5py' does not work, relating to old-version MATLAB file.
    p_thre : float, optional and should be within range [0, 1)    
        The threshold to initialize PSameList class.
        The default value is 0.5
    
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
        return PSameList(p_same, p_thre=p_thre)
    elif open_type == 'scipy':
        raise NotImplementedError("open type 'scipy' is not supported yet")
    else:
        raise ValueError("open type should be 'h5py' or 'scipy'")

def read_matlab_data(reference, file, dtype: str = 'int'):
    """
    Recursively read data from an HDF5 reference in a MATLAB file.

    Parameters:
    reference (h5py.Reference): HDF5 reference to a dataset or a group.
    file (h5py.File): HDF5 file object.
    dtype (str): Data type of the data to be read.
        By default, 'int'. Other options: 'float'.

    Returns:
    data: The data stored in the dataset or group.
    """
    # Dereference the HDF5 object
    obj = file[reference]

    # If the object is a dataset, return its contents
    if isinstance(obj, h5py.Dataset):
        temp = obj[()]
        if temp.shape[0] == 2:
            return []
        
        if isinstance(temp, np.ndarray) and (temp.dtype == np.int64 or temp.dtype == np.float64) and dtype == 'int':
            return temp.astype(np.int64)[0]
        elif isinstance(temp, np.ndarray) and (temp.dtype == np.int64 or temp.dtype == np.float64) and dtype == 'float':
            return temp.astype(np.float64).T[0]
        else:
            return temp
    # If the object is a group (like a MATLAB cell), read its contents recursively
    elif isinstance(obj, h5py.Group):
        return [read_matlab_data(ref, file) for ref in obj]  

def read_all_to_all_indexes(
    dir_name: str, 
    open_type = 'h5py'
) -> AllToAll:
    """read_all_to_all_indexes: Read the all-to-all indice from the MATLAB file saved by CellReg.

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
            dataset_a = modeled_data_struct['all_to_all_indexes']

            # Initialize an empty list to hold the fully dereferenced data
            fully_dereferenced_data = []

            for i, ref in enumerate(dataset_a):
                # Iterate for n_sessions. Set session i as reference session.
                all_to_all_oneday = read_matlab_data(ref[0], f, dtype='int')
                # Length of nested_data: n_neuron
                sessions_wise_data = []
                for session_ref in all_to_all_oneday:
                    # The ref session compares with the remaining. Iterates for n_session times. 
                    A_to_B_indexes = []
                    for neuron_ref in session_ref:
                        # Iterates for n_neuron times, dereference each neuron's potential partners.

                        # Further dereference if the data contains more HDF5 references
                        res = list(read_matlab_data(neuron_ref, f, dtype='int'))
                        A_to_B_indexes.append(res)
                        
                    sessions_wise_data.append(A_to_B_indexes)
                
                fully_dereferenced_data.append(sessions_wise_data)
        return AllToAll(fully_dereferenced_data)

def read_all_to_all_psame(
    dir_name: str, 
    open_type: str = 'h5py', 
    p_model: str = 'spatial_correlation_model'
) -> AllToAll:
    """read_all_to_all_psame: Read the all-to-all indice from the MATLAB file saved by CellReg.

    Parameters
    ----------
    dir_name : str
        The directory of the MATLAB file
    open_type : str, optional
        The method to open the MATLAB file, depending on the file signature
        Generally, 'h5py' and 'scipy' are used, by default 'h5py'
        'h5py' is needed when using v7.3 signature to save the file
        'scipy' is needed when 'h5py' does not work, relating to old-version MATLAB file.
    p_model : str, optional
        The model of psame, by default 'spatial_correlation_model'. You could select
        'centroid_distance_model' as well.
            
    Returns
    -------
    all_to_all_psame, very complex list hierarchy.
    
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
            dataset_a = modeled_data_struct['all_to_all_p_same_'+p_model]

            # Initialize an empty list to hold the fully dereferenced data
            fully_dereferenced_data = []

            for i, ref in enumerate(dataset_a):
                # Iterate for n_sessions. Set session i as reference session.
                all_to_all_oneday = read_matlab_data(ref[0], f, dtype='float')
                # Length of nested_data: n_neuron
                sessions_wise_data = []
                for session_ref in all_to_all_oneday:
                    # The ref session compares with the remaining. Iterates for n_session times. 
                    A_to_B_indexes = []
                    for neuron_ref in session_ref:
                        # Iterates for n_neuron times, dereference each neuron's potential partners.

                        # Further dereference if the data contains more HDF5 references
                        res = list(read_matlab_data(neuron_ref, f, dtype='float'))
                        A_to_B_indexes.append(res)
                        
                    sessions_wise_data.append(A_to_B_indexes)
                
                fully_dereferenced_data.append(sessions_wise_data)
        return AllToAll(fully_dereferenced_data)

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
    import h5py
    import time
    import sys
    # Attempting to read the contents of the dataset 'a' with corrected handling of references

    dir_name = r"E:\Data\Cross_maze\10227\Super Long-term Maze 1\Cell_reg\modeled_data_struct.mat"
    t1 = time.time()
    a = read_all_to_all_psame(dir_name=dir_name) #read_all_to_all_indixes(dir_name=dir_name)
    print(time.time() - t1)
    # Examine the memory taken by 
    print(sys.getsizeof(a))
    print(a[0, 1, 0])

    
    # Test
    """    
    dir_name = r"E:\Data\Cross_maze\10227\Super Long-term Maze 1\Cell_reg\cellRegistered.mat"
    # Test
    index_map = read_index_map(dir_name)
    print("index_map:", type(index_map), index_map.shape)
    
    register_score = read_register_score(dir_name)
    print("register_score:", type(register_score), register_score.shape)
    
    exclusivity_score = read_exclusivity_score(dir_name)
    print("exclusivity_score:", type(exclusivity_score), exclusivity_score.shape)
    
    psame = read_psame(dir_name)
    print("psame:", type(psame), psame.shape)
    """

