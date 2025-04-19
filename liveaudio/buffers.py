import numpy as np
from threading import Lock

class CircularBuffer:
    """Circular Buffer for real-time audio processing.

    This class implements an efficient circular buffer designed for audio processing
    applications where data arrives in small chunks (hop-sized) and needs to be 
    processed in larger frames. It accumulates input data until a complete frame
    is available.
    
    The buffer collects 'r' hop-sized chunks (where r = frame_size/hop_size) before
    marking itself as full. Once full, it can provide the entire frame of data in
    chronological order, properly handling the circular wrap-around.
    
    Features:
    - Efficient storage with minimal copying
    - Thread-safe operations (optional)
    - Automatic tracking of buffer fullness
    - Returns contiguous data arrays for optimal performance
    
    Typical usage:
        buffer = CircularBuffer(frame_size, hop_size)
        # Push hop-sized chunks until buffer is full
        for chunk in input_chunks:
            buffer.push(chunk)
            if buffer.full:
                frame = buffer.get()
                # process the complete frame
    """
    def __init__(self, size, hop, dtype = np.float64, threadSafe=True):
        """Initialize a high-performance circular buffer for audio processing.
        
        Args:
            size: Size of the buffer in samples (must be divisible by hop)
            hop: Size of data chunks for updates (hop size)
            dtype: NumPy data type for the buffer (default: np.float64)
            threadSafe: Whether to use thread-safe operations (default: True)
        
        Raises:
            AssertionError: If size is not divisible by hop
        """
        assert size % hop == 0, "Size must be divisible by hop size"
        self.size = size
        self.hop = hop
        self.r = size // hop
        self.full = False
        
        # Create buffer for data storage
        self.buffer = np.zeros(size, dtype=dtype)
        self.position = 0
        
        
        # Thread safety
        self._threadSafe = threadSafe
        if threadSafe:
            self._lock = Lock()

    def push(self, data):
        """Push a chunk of data into the circular buffer.
        
        This method adds a hop-sized chunk of data to the circular buffer,
        automatically handling the circular wrapping when the buffer end is reached.
        
        Args:
            data: 1D numpy array of size 'hop' containing the input chunk
            
        Returns:
            self: The buffer object for method chaining
            
        Raises:
            TypeError: If input is not a numpy array
            ValueError: If input is not 1D or has incorrect size
        """
        # Validate input
        if not isinstance(data, np.ndarray):
            raise TypeError("Input must be a NumPy array")
        if data.ndim != 1:
            raise ValueError("Input must be a 1D NumPy array")
        if data.size != self.hop:
            raise ValueError(f"Input array size must be {self.hop}")
        
        # Thread safety
        if self._threadSafe:
            with self._lock:
                self._pushImpl(data)
        else:
            self._pushImpl(data)
        
        return self
    
    def _pushImpl(self, data):
        """Internal implementation of the push operation"""
        self.buffer[self.position:self.position+self.hop] = data
        self.position += self.hop
        if self.position >= self.size:
            self.position = 0
            self.full = True
    
    def __len__(self):
        """Return buffer length"""
        return self.size
    
    def get(self):
        """Get all buffered data as a single contiguous array in chronological order.
        
        This method retrieves the entire buffer contents arranged in proper
        time sequence, handling the circular nature of the buffer internally.
        Always returns a contiguous copy for consistent performance.
        
        Returns:
            A 1D numpy array containing all valid data in the buffer. If the buffer
            is not yet full, only returns the data that has been pushed so far.
        """
        if self._threadSafe:
            with self._lock:
                return self._getImpl()
        else:
            return self._getImpl()
            
    def _getImpl(self):
        """Implementation of the get operation"""
        # If buffer isn't full yet, only return data that's been pushed
        if not self.full:
            return self.buffer[:self.position].copy()
        
        # Data wraps around the end of the buffer
        if self.position == 0:
            return self.buffer.copy()

        return np.concatenate((self.buffer[self.position:], self.buffer[:self.position]))
    
class OACBuffer:
    """Overlap-Add Circular Buffer for real-time audio processing.
    
    This class implements a specialized circular buffer for overlap-add processing,
    commonly used in audio applications like vocoders, phase vocoders, and spectral
    processors. It manages the storage, alignment, and summation of overlapping frames.
    
    The buffer stores 'r' frames (where r = frame_size/hop_size) and performs 
    overlap-add operations to produce hop-sized output chunks. Each new frame 
    contributes to multiple consecutive output hops.
    
    Features:
    - Automatic frame alignment for overlap-add processing
    - Optional windowing during frame storage
    - Thread-safe operations (optional)
    - Efficient hop-by-hop output generation
    
    Typical usage:
        buffer = OACBuffer(frame_size, hop_size, window=hann_window)
        for frame in frames:
            output_hop = buffer.pushGet(frame)
            # process output_hop
    """    
    def __init__(self, size, hop, window = None, dtype = np.float64, threadSafe=True):
        """Initialize an Overlap-Add Circular Buffer.
        
        Args:
            size: Size of the buffer in samples (must be divisible by hop)
            hop: Size of data chunks for updates (hop size)
            window: Optional window function array of length 'size' to apply to frames
            dtype: NumPy data type for the buffer (default: np.float64)
            threadSafe: Whether to use thread-safe operations (default: True)
        """
        assert size % hop == 0, "Size must be divisible by hop size"
        assert window is None or len(window) == size, "Window size must be size"
        self.size = size
        self.hop = hop
        self.window = window
        self.r = size // hop
        
        # Create buffer for data storage
        self.buffer = np.zeros((self.r,size), dtype=dtype)
        self.position = 0
        self.n = 0
        
        # Thread safety
        self._threadSafe = threadSafe
        if threadSafe:
            self._lock = Lock()

    def pushGet(self, data):
        """Push a frame of data into the buffer and return the next processed hop-sized chunk.
        
        This method performs overlap-add processing by:
        1. Applying the window function (if provided) to the input frame
        2. Storing the frame in the buffer with proper alignment
        3. Computing the sum of overlapping frame segments
        4. Returning the current hop-sized output segment
        
        Args:
            data: 1D numpy array of length 'size' containing the input frame
            
        Returns:
            A hop-sized 1D numpy array containing the sum of overlapping frame segments
            
        Raises:
            TypeError: If input is not a numpy array
            ValueError: If input is not 1D or has incorrect size
        """
        # Validate input
        if not isinstance(data, np.ndarray):
            raise TypeError("Input must be a NumPy array")
        if data.ndim != 1:
            raise ValueError("Input must be a 1D NumPy array")
        if data.size != self.size:
            raise ValueError(f"Input array size must be {self.size}")
        
        # Thread safety
        if self._threadSafe:
            with self._lock:
                return self._pushGetImpl(data)
        else:
            return self._pushGetImpl(data)
    
    def _pushGetImpl(self, data):
        """Internal implementation of the push operation"""
        self.buffer[self.n,:] = np.roll(data if self.window is None else data*self.window, self.position)
        result = self.buffer[:, self.position: self.position+self.hop].sum(0)
        self.position += self.hop
        self.n += 1
        if self.position >= self.size: self.n = self.position = 0
        return result
    
    def __len__(self):
        """Return buffer length"""
        return self.size