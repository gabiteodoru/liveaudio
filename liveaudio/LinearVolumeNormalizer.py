import numpy as np

class LinearVolumeNormalizer:
    """
    A class for normalizing audio signals by applying a linear gain function.
    
    This normalizer maximizes volume of audio chunks by applying a linearly increasing 
    gain function while ensuring the signal never exceeds a specified amplitude limit.
    It's designed to be used with streaming audio by maintaining state between chunks.
    
    The normalizer works by finding the steepest possible linear gain function that won't
    cause any sample to exceed the limit. The gain function starts at the previous chunk's
    ending gain value and increases linearly throughout the current chunk.
    
    IMPORTANT: This normalizer cannot guarantee 100% that audio won't exceed the limit
    because it cannot change the first sample of the next chunk (as this would cause
    a discontinuity in the audio). For this reason, it's recommended to allow some 
    headroom by setting the limit below 1.0 (default is 0.95).
    
    Attributes:
        limit (float): Maximum absolute amplitude allowed in the normalized signal (0.0-1.0).
        v0 (float): Initial gain value for the first chunk; updated after each chunk.
    """
    
    def __init__(self, limit=.95, v0=.95):
        """
        Initialize the LinearVolumeNormalizer with amplitude limit and starting gain.
        
        Parameters:
            limit (float): Maximum absolute amplitude allowed in the normalized signal, 
                           typically between 0.0 and 1.0. Default is 0.95.
            v0 (float): Initial gain value for the first chunk. Default is 0.95.
        """
        self.limit = limit
        self.v0 = v0
        
    def normalize(self, audio):
        """
        Normalize an audio chunk using a linear gain function.
        
        This method calculates the maximum possible linear gain increase that can be
        applied to the audio chunk without any sample exceeding the specified amplitude
        limit. The gain starts at the current v0 value (carried over from previous chunks)
        and increases linearly across the chunk.
        
        The method handles various edge cases including empty arrays, arrays with all zeros,
        and negative audio values. It uses absolute values for calculations to ensure
        both positive and negative peaks are properly considered.
        
        IMPORTANT: This method cannot guarantee 100% that subsequent chunks won't exceed 
        the limit because it cannot modify the first sample of the next chunk (as this would 
        cause a discontinuity). This is why the default limit is set to 0.95 rather than 1.0,
        to provide some safety headroom.
        
        Parameters:
            audio (numpy.ndarray): A 1D numpy array containing the audio samples to normalize.
                                  Values are typically in the range [-1.0, 1.0].
                                  
        Returns:
            numpy.ndarray: The normalized audio with the linear gain function applied.
            
        Notes:
            - The gain function is calculated as: v(t) = v0 + k*t, where k is the slope
              of the gain function and t is the sample index.
            - After processing, v0 is updated to the ending gain value for continuity 
              with the next chunk.
            - For silent audio (all zeros), the original array is returned unchanged.
        """
        T = len(audio)
        if T == 0: return audio
        absAudio = np.abs(audio) # Use absolute values to handle negative samples
        nonZeroIndices = np.where(absAudio[1:] > 0)[0] + 1 # Find non-zero samples to avoid division by zero
        if len(nonZeroIndices) == 0: return audio # No non-zero samples after the first one
        kMax = (self.limit - self.v0) / T # Calculate k (gain slope) to ensure we don't exceed limit
        sampleFactors = (self.limit - self.v0 * (aa:=absAudio[nonZeroIndices])) / (aa * nonZeroIndices) # Calculate gain factors for each non-zero sample
        k = np.min(np.append(kMax, np.min(sampleFactors))) # Find minimum k value that ensures no sample exceeds the limit
        v = self.v0 + k * np.arange(T) # Generate linear gain function
        self.v0 = self.v0 + k * T # Update starting gain for next chunk
        return audio * v