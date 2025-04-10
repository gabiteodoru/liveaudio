import numba
import numpy as np


'''
Fast, forward-only Viterbi
Includes versions optimized for sparseness and Toeplitz / Block-Topelitz structure
Both pure max-product, as well as past-sum-product+present-max
Computation is half-staggered i.e. 
  first apply output distribution, then apply transition to prepare for next time step
'''

@numba.jit(nopython=True, cache=True)
def onlineViterbiState(value: np.ndarray, log_prob: np.ndarray, log_trans: np.ndarray):
    value = value+log_prob
    result = np.argmax(value)
    trans_out = value[:, np.newaxis] + log_trans
    max_vals = np.zeros(trans_out.shape[1])
    for j in range(trans_out.shape[1]):
        max_vals[j] = np.max(trans_out[:, j])
    return max_vals, result, value

@numba.jit(nopython=True, cache=True, parallel=True)
def onlineViterbiStateOpt(value: np.ndarray, log_prob: np.ndarray, log_trans_row: np.ndarray):
    value = value+log_prob
    result = np.argmax(value)

    halfWidth = (len(log_trans_row) - 1) // 2
    n_states = len(value)
    max_vals = np.zeros(n_states)
    
    # For each destination state - use parallel range
    for j in numba.prange(n_states):
        # Determine the range of source states that can transition to state j
        start_idx = max(0, j - halfWidth)
        end_idx = min(n_states, j + halfWidth + 1)

        curr_max = -np.inf
        for i in range(start_idx, end_idx):
            trans_offset = halfWidth + j - i
            trans_val = value[i] + log_trans_row[trans_offset]
            curr_max = max(curr_max, trans_val)
        max_vals[j] = curr_max
    return max_vals, result



@numba.jit(nopython=True, cache=True, parallel=True)
def blockViterbiStateOpt(value: np.ndarray, log_prob: np.ndarray, 
                         log_trans_00: np.ndarray, log_trans_01: np.ndarray,
                         log_trans_10: np.ndarray, log_trans_11: np.ndarray,
                         n1: int, correspondence: np.ndarray):
    """
    Optimized block-structured Viterbi algorithm for real-time applications.
    
    This function implements a forward-only Viterbi algorithm that takes advantage of sparse,
    Toeplitz transition matrices structured as a 2x2 block matrix. It applies the current
    observation probabilities and computes the next state probabilities, ready for the next call.
    
    Parameters:
    ----------
    value : np.ndarray
        Current log probabilities for each state.
    log_prob : np.ndarray
        Log probabilities of current observations for each state.
    log_trans_00 : np.ndarray
        Band of the transition matrix for group 0 → group 0 transitions.
        Should be of odd length (2*halfWidth + 1), centered at the diagonal.
    log_trans_01 : np.ndarray
        Band of the transition matrix for group 0 → group 1 transitions.
        Should be of odd length, centered at the correspondence point.
    log_trans_10 : np.ndarray
        Band of the transition matrix for group 1 → group 0 transitions.
        Should be of odd length, centered at the correspondence point.
    log_trans_11 : np.ndarray
        Band of the transition matrix for group 1 → group 1 transitions.
        Should be of odd length, centered at the diagonal.
    n1 : int
        Number of states in the first group (group 0).
    correspondence : np.ndarray
        Array mapping each state to its corresponding state in the other group.
        For indices 0 to n1-1 (group 0): contains local indices (0-based) in group 1.
        For indices n1 to n1+n2-1 (group 1): contains local indices (0-based) in group 0.
        Used to determine where to center the band for between-group transitions.
        The index into this array is the destination state (j), and the value 
        retrieved is the corresponding source state in the other group that serves
        as the center of the transition band.
        
    Returns:
    -------
    tuple
        A tuple containing:
        - max_vals: np.ndarray - The updated log probabilities for each state
        - result: int - The index of the most likely state after applying log_prob
    
    Notes:
    -----
    - The transition matrices are represented as bands to take advantage of their sparse
      Toeplitz structure, where transitions depend only on the distance between states.
    - For within-group transitions (00 and 11), the band is centered on the diagonal.
    - For between-group transitions (01 and 10), the band is centered on the corresponding
      state as defined by the correspondence array.
    - The half-width of each band is calculated as (len(band) - 1) // 2, so bands should
      have odd lengths for symmetric spanning around the center point.
    - The algorithm is optimized for Numba parallel execution.
    - This implementation is intended for situations where the range of states 
      in each group is the same, but at different resolutions.
    """    
    # Update state values with current observation probabilities
    value = value+log_prob
    
    # Find the most likely state
    result = np.argmax(value)
    
    # Pre-compute constants
    n_states = len(value)
    n2 = n_states - n1
    
    # Pre-compute half-widths for each transition matrix
    halfWidth_00 = (len(log_trans_00) - 1) // 2
    halfWidth_01 = (len(log_trans_01) - 1) // 2
    halfWidth_10 = (len(log_trans_10) - 1) // 2
    halfWidth_11 = (len(log_trans_11) - 1) // 2
    
    # Prepare the output values
    max_vals = np.zeros(n_states)
    
    # For each destination state - use parallel range
    for j in numba.prange(n_states):
        # Initialize with an extremely low value
        curr_max = -np.inf
        
        # Get corresponding state in the other group
        corresponding_i = correspondence[j]
        
        if j < n1:  # Destination is in group 0
            # Consider transitions from group 0 (via matrix 00)
            start_idx_0 = max(0, j - halfWidth_00)
            end_idx_0 = min(n1, j + halfWidth_00 + 1)
            
            # Find max transition from group 0 to j
            for i in range(start_idx_0, end_idx_0):
                trans_offset = halfWidth_00 + j - i
                trans_val = value[i] + log_trans_00[trans_offset]
                curr_max = max(curr_max, trans_val)
                
            # Consider transitions from group 1 (via matrix 10)
            # Center the band around this corresponding state
            start_idx_1 = max(0, corresponding_i - halfWidth_10)
            end_idx_1 = min(n2, corresponding_i + halfWidth_10 + 1)
            
            # Find max transition from group 1 to j
            for i in range(start_idx_1, end_idx_1):
                # Adjust i to global index (add n1)
                i_global = i + n1
                trans_offset = halfWidth_10 + corresponding_i - i
                trans_val = value[i_global] + log_trans_10[trans_offset]
                curr_max = max(curr_max, trans_val)
                
        else:  # Destination is in group 1
            # Adjust j to local index in group 1
            j_local = j - n1
            
            # Consider transitions from group 0 (via matrix 01)
            # Center the band around this corresponding state
            start_idx_0 = max(0, corresponding_i - halfWidth_01)
            end_idx_0 = min(n1, corresponding_i + halfWidth_01 + 1)
            
            # Find max transition from group 0 to j
            for i in range(start_idx_0, end_idx_0):
                trans_offset = halfWidth_01 + corresponding_i - i
                trans_val = value[i] + log_trans_01[trans_offset]
                curr_max = max(curr_max, trans_val)
                
            # Consider transitions from group 1 (via matrix 11)
            start_idx_1 = max(0, j_local - halfWidth_11)
            end_idx_1 = min(n2, j_local + halfWidth_11 + 1)
            
            # Find max transition from group 1 to j
            for i in range(start_idx_1, end_idx_1):
                # Adjust i to global index (add n1)
                i_global = i + n1
                trans_offset = halfWidth_11 + j_local - i
                trans_val = value[i_global] + log_trans_11[trans_offset]
                curr_max = max(curr_max, trans_val)
        
        max_vals[j] = curr_max
    
    return max_vals, result

@numba.jit(nopython=True, cache=True)
def sumProductViterbi(value: np.ndarray, log_prob: np.ndarray, log_trans: np.ndarray):
    value += log_prob
    result = np.argmax(value)
    trans_out = value[:, np.newaxis] + log_trans
    # Log-sum-exp trick with broadcasting
    max_vals = np.zeros(trans_out.shape[1])
    for i in range(trans_out.shape[1]):
        max_vals[i] = np.max(trans_out[:, i])
    exp_diff = np.exp(trans_out - max_vals)  # Broadcasting subtracts max_vals from each column
    exp_sum = np.sum(exp_diff, axis=0)
    value = log_prob + max_vals + np.log(exp_sum)
    return value, result

