import numpy as np
from scipy.optimize import minimize_scalar
square = lambda x:x*x

class LinearVolumeNormalizer:
    """
    A class for normalizing audio signals by applying a linear gain function.
    
    This normalizer maximizes volume of audio chunks by applying a linearly increasing 
    gain function while ensuring the signal never exceeds a specified amplitude limit.
    It's designed to be used with streaming audio by maintaining state between chunks.
    
    The normalizer works by finding the optimal linear gain function through constrained
    optimization. It balances maximizing overall signal volume while applying a penalty
    for exceeding the amplitude limit. The gain function starts at the previous chunk's
    ending gain value and increases linearly throughout the current chunk.
    
    IMPORTANT: This normalizer cannot guarantee 100% that audio won't exceed the limit
    because it cannot change the first sample of the next chunk (as this would cause
    a discontinuity in the audio). For this reason, it's recommended to allow some 
    headroom by setting the limit below 1.0 (default is 0.95).
    
    Attributes:
        limit (float): Maximum absolute amplitude allowed in the normalized signal (0.0-1.0).
        v0 (float): Initial gain value for the first chunk; updated after each chunk.
        maxKT (float): Maximum allowed gain increase over the entire chunk (default 0.2).
        lmbda (float): Penalty factor for samples exceeding the limit (default 0.1).
    """
    
    def __init__(self, limit=.95, v0=.95, maxKT = .2, lmbda = .1):
        """
        Initialize the LinearVolumeNormalizer with amplitude limit and starting gain.
        
        Parameters:
            limit (float): Maximum absolute amplitude allowed in the normalized signal, 
                          typically between 0.0 and 1.0. Default is 0.95.
            v0 (float): Initial gain value for the first chunk. Default is 0.95.
            maxKT (float): Maximum allowed gain increase over the entire chunk. Default is 0.2.
            lmbda (float): Penalty factor for samples exceeding the limit. Higher values enforce
                          stricter adherence to the limit. Default is 0.1.
        """
        self.limit = limit
        self.v0 = v0
        self.maxKT = maxKT
        self.lmbda = lmbda
        
    def normalize(self, x):
        """
        Normalize an audio chunk using an optimized linear gain function.
        
        This method uses constrained optimization to find the best linear gain increase
        that can be applied to the audio chunk. It maximizes overall signal volume while
        applying a penalty for any sample exceeding the specified amplitude limit.
        
        The optimization is performed using scipy's minimize_scalar with the 'bounded' method,
        which efficiently finds the optimal gain slope (k) that balances volume maximization
        with limit adherence.
        
        The method handles various edge cases including empty arrays and arrays with all zeros.
        It uses squared values for calculations to ensure both positive and negative peaks 
        are properly considered.
        
        Parameters:
            x (numpy.ndarray): A 1D numpy array containing the audio samples to normalize.
                              Values are typically in the range [-1.0, 1.0].
                              
        Returns:
            numpy.ndarray: The normalized audio with the optimized linear gain function applied.
            
        Notes:
            - The gain function is calculated as: v(t) = v0 + k*t, where k is the slope
              of the gain function and t is the sample index.
            - The optimization uses a cost function that maximizes signal power while 
              applying a penalty for samples exceeding the limit.
            - After processing, v0 is updated to the ending gain value for continuity 
              with the next chunk.
            - For silent audio (all zeros), the original array is returned unchanged.
            - The function prints the maximum absolute value of the normalized audio.
        """
        T = len(x)
        if T == 0: return x
        x2 = square(x) # Use absolute values to handle negative samples
        t = np.where(x2[1:] > 0)[0] + 1 # Find non-zero samples to avoid division by zero
        if len(t) == 0: return x # No non-zero samples after the first one
        lmbda = square(T*self.lmbda)
        v0, L = self.v0, self.limit
        x2 = x2[t]
        costF = lambda k: -(q2:=x2*square(v0+k*t)).sum() + lmbda * np.maximum(0, q2-square(L)).sum()
        k = minimize_scalar(costF, bounds=(-v0/T, self.maxKT/T), method='bounded', options={'xatol': 1e-6, 'maxiter':8}).x
        v = self.v0 + k * np.arange(T) # Generate linear gain function
        self.v0 = self.v0 + k * T # Update starting gain for next chunk
        return x * v