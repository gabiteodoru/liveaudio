import numpy as np
from scipy.optimize import minimize_scalar
square = lambda x:x*x
import numba

@numba.njit(cache = True)
def loss(k, x2, v0, t, lmbda, L2):
    """
    Calculate the optimization loss function for audio normalization.
    
    This function computes the loss value for a given gain slope (k), balancing two objectives:
    - Maximizing the overall signal power (negative term)
    - Penalizing any samples that exceed the amplitude limit (positive term)
    
    Parameters:
        k (float): The gain slope to evaluate
        x2 (numpy.ndarray): Squared audio samples (pre-computed for efficiency)
        v0 (float): Initial gain value 
        t (numpy.ndarray): Time indices for each sample
        lmbda (float): Penalty factor for exceeding the limit
        L2 (float): Squared limit value
        
    Returns:
        float: The loss value for the given k
    """    
    q = (v0+k*t)
    q2 = x2*q*q
    return -q2.sum()+lmbda * np.maximum(0, q2-L2).sum()

@numba.njit(cache=True)
def fast_bounded_optimize_loss(x2, v0, t, lmbda, L2, bounds, maxiter=8, tol=1e-6):
    """
    Optimizes the gain slope (k) using golden section search.
    
    This is a JIT-compiled implementation of golden section search optimization,
    specifically tailored for audio normalization. It finds the optimal gain slope
    that maximizes volume while respecting amplitude limits.
    
    Parameters:
        x2 (numpy.ndarray): Squared audio samples
        v0 (float): Initial gain value
        t (numpy.ndarray): Time indices for each sample
        lmbda (float): Penalty factor for exceeding the limit
        L2 (float): Squared limit value
        bounds (tuple): Lower and upper bounds for k (min, max)
        maxiter (int): Maximum number of optimization iterations
        tol (float): Convergence tolerance
        
    Returns:
        float: The optimal gain slope (k)
    """
    a, b = bounds
    gr = (np.sqrt(5) + 1) / 2  # Golden ratio
    
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    
    fc, fd = loss(c, x2, v0, t, lmbda, L2), loss(d, x2, v0, t, lmbda, L2)
    
    for _ in range(maxiter):
        if fc < fd:  # For minimization
            b, d, fd = d, c, fc
            c = b - (b - a) / gr
            fc = loss(c, x2, v0, t, lmbda, L2)
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) / gr
            fd = loss(d, x2, v0, t, lmbda, L2)
            
        if abs(b - a) < tol * (abs(b) + abs(a)):
            break
            
    return (a + b) / 2
    

class LinearVolumeNormalizer:
    """
    A class for normalizing audio signals by applying a linear gain function.
    
    This normalizer maximizes volume of audio chunks by applying a linearly increasing 
    gain function while ensuring the signal never exceeds a specified amplitude limit.
    It's designed to be used with streaming audio by maintaining state between chunks.
    
    The normalizer works by finding the optimal linear gain function through constrained
    optimization. It balances maximizing overall signal volume while applying a penalty
    for exceeding the amplitude limit. For enhanced efficiency, the audio is segmented 
    into blocks and only the maximum value from each block is used for optimization.
    
    IMPORTANT: This normalizer cannot guarantee 100% that audio won't exceed the limit
    because it cannot change the first sample of the next chunk (as this would cause
    a discontinuity in the audio). For this reason, it's recommended to allow some 
    headroom by setting the limit below 1.0 (default is 0.95).
    
    Attributes:
        limit (float): Maximum absolute amplitude allowed in the normalized signal (0.0-1.0).
        v0 (float): Initial gain value for the first chunk; updated after each chunk.
        maxKT (float): Maximum allowed gain increase over the entire chunk.
        lmbda (float): Penalty factor for samples exceeding the limit.
        nBlocks (int): Number of blocks to divide the audio into for optimization.
    """
    
    def __init__(self, limit=.95, v0=.95, maxKT = .2, lmbda = .1, nBlocks = 32):
        """
        Initialize the LinearVolumeNormalizer with amplitude limit and starting gain.
        
        Parameters:
            limit (float): Maximum absolute amplitude allowed in the normalized signal, 
                          typically between 0.0 and 1.0. Default is 0.95.
            v0 (float): Initial gain value for the first chunk. Default is 0.95.
            maxKT (float): Maximum allowed gain increase over the entire chunk. Default is 0.2.
            lmbda (float): Penalty factor for samples exceeding the limit. Higher values enforce
                          stricter adherence to the limit. Default is 0.1.
            nBlocks (int): Number of blocks to divide the audio into for optimization. Default is 32.
                          Higher values provide more precision but slower performance.
        """
        self.limit = limit
        self.v0 = v0
        self.maxKT = maxKT
        self.lmbda = lmbda
        self.nBlocks = nBlocks
        
    def normalize(self, x):
        """
        Normalize an audio chunk using an optimized linear gain function.
        
        This method uses JIT-compiled golden section search optimization to find the best
        linear gain increase that can be applied to the audio chunk. It maximizes overall 
        signal volume while applying a penalty for any sample exceeding the specified amplitude limit.
        
        For performance optimization, the audio is divided into blocks, and only the maximum value
        from each block is used in the optimization process.
        
        Parameters:
            x (numpy.ndarray): A 1D numpy array containing the audio samples to normalize.
                              Values are typically in the range [-1.0, 1.0].
                              
        Returns:
            numpy.ndarray: The normalized audio with the optimized linear gain function applied.
            
        Notes:
            - The gain function is calculated as: v(t) = v0 + k*t, where k is the slope
              of the gain function and t is the sample index.
            - The method handles various edge cases including empty arrays and arrays with all zeros.
            - After processing, v0 is updated to the ending gain value for continuity 
              with the next chunk.
        """
        T = len(x)
        if T == 0: return x
        x2 = square(x) # Use absolute values to handle negative samples
        if (x2<1e-8).all(): return x
        blRng = np.linspace(0, len(x2), self.nBlocks+1, dtype=int)
        k = fast_bounded_optimize_loss(np.maximum.reduceat(x2, blRng[:-1]), self.v0, (blRng[:-1]+blRng[1:]) / 2, square(T*self.lmbda), square(self.limit), bounds=(-self.v0/T, self.maxKT/T), tol= 1e-8, maxiter=15)
        v = self.v0 + k * np.arange(T) # Generate linear gain function
        self.v0 = self.v0 + k * T # Update starting gain for next chunk
        return x * v

def _precompile_numba_functions():
    fast_bounded_optimize_loss(np.random.randn(32).astype(np.float64), 1., np.arange(32,dtype=np.float64), .01, 1., (-1,1), maxiter=8, tol=1e-6)
_precompile_numba_functions()    
