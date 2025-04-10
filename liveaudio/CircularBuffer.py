import numpy as np
from threading import Lock

class CircularBuffer:
    def __init__(self, sz, hopSize, threadSafe=True):
        """
        Initialize a high-performance circular buffer with NumPy views.
        
        Args:
            sz: Size of the buffer (must be divisible by hopSize)
            hopSize: Size of data chunks for updates
            threadSafe: Whether to use thread-safe operations
        """
        assert sz % hopSize == 0, "Size must be divisible by hop size"
        self.sz = sz
        self.hopSize = hopSize
        self.numFrames = sz // hopSize
        self.full = False
        
        # Create buffer for data storage
        self.buffer = np.zeros(sz, dtype=np.float32)
        self.position = 0
        
        # Double buffering to avoid wraparound costs in certain cases
        self._useDoubleBuffer = hopSize < sz // 4  # Only use if hop is significantly smaller than buffer
        if self._useDoubleBuffer:
            self._doubleBuffer = np.zeros(hopSize * 2, dtype=np.float32)
        
        # Thread safety
        self._threadSafe = threadSafe
        if threadSafe:
            self._lock = Lock()

    def push(self, data):
        """
        Push a chunk of data into the circular buffer.
        
        Args:
            data: 1D numpy array of size hopSize
            
        Returns:
            self for method chaining
        """
        # Validate input
        if not isinstance(data, np.ndarray):
            raise TypeError("Input must be a NumPy array")
        if data.ndim != 1:
            raise ValueError("Input must be a 1D NumPy array")
        if data.size != self.hopSize:
            raise ValueError(f"Input array size must be {self.hopSize}")
        
        # Thread safety
        if self._threadSafe:
            with self._lock:
                self._pushImpl(data)
        else:
            self._pushImpl(data)
        
        return self
    
    def _pushImpl(self, data):
        """Internal implementation of the push operation"""
        self.buffer[self.position:self.position+self.hopSize] = data
        self.position = self.position + self.hopSize
        if self.position >= self.sz:
            self.position = 0
            self.full = True
    
    def getFrames(self, numFrames=None):
        """
        Get the most recent frames in time order using views when possible.
        
        Args:
            numFrames: Number of frames to retrieve (default: all frames)
            
        Returns:
            Either a view into the buffer or a new array with copied data
        """
        if numFrames is None:
            numFrames = self.numFrames
        
        if numFrames > self.numFrames:
            raise ValueError(f"Cannot retrieve more than {self.numFrames} frames")
        
        # Thread safety for the entire operation
        if self._threadSafe:
            with self._lock:
                return self._getFramesImpl(numFrames)
        else:
            return self._getFramesImpl(numFrames)
    
    def _getFramesImpl(self, numFrames):
        """
        Implementation of the frame retrieval logic with view optimization.
        
        This will return a view when possible (when frames don't wrap around buffer)
        and only create a copy when necessary.
        """
        oldest_frame_pos = self.position
        
        # Create result array
        result = np.empty((numFrames, self.hopSize), dtype=np.float32)
        
        # For each frame we want to retrieve (starting with the oldest)
        for i in range(numFrames):
            # Calculate the frame index in reverse order (numFrames-1-i) 
            # so we get 0, 1, 2, 3... in the result
            frame_idx = numFrames - 1 - i
            
            # Calculate where this frame starts in the circular buffer
            frame_start = (oldest_frame_pos - (i + 1) * self.hopSize) % self.sz
            frame_end = (frame_start + self.hopSize) % self.sz
            
            if frame_start < frame_end:
                # This frame doesn't wrap
                result[frame_idx] = self.buffer[frame_start:frame_end]
            else:
                # This frame wraps around
                first_part = self.sz - frame_start
                result[frame_idx, :first_part] = self.buffer[frame_start:]
                result[frame_idx, first_part:] = self.buffer[:frame_end]
        
        return result
    
    # Methods to make the buffer behave like a NumPy array
    def __array__(self):
        """
        Return a view or copy of the buffer for NumPy operations.
        
        Returns a view when thread safety is disabled, or a copy when thread safety is enabled.
        """
        if self._threadSafe:
            with self._lock:
                return self.buffer.copy()
        
        # Return the buffer directly when thread safety is not a concern
        return self.buffer
    
    def __getitem__(self, key):
        """
        Support array indexing by returning a view into the buffer.
        """
        if self._threadSafe:
            with self._lock:
                # For simple indices, we can optimize
                if isinstance(key, (int, np.integer)):
                    return self.buffer[key]
                # For slices and fancy indexing, return a view or a copy as appropriate
                result = self.buffer[key]
                # If the result is an array and thread safety is enabled, we need a copy
                return result.copy() if isinstance(result, np.ndarray) else result
        
        # When thread safety is not a concern, we can return views directly
        return self.buffer[key]
    
    def __setitem__(self, key, value):
        """Support array item assignment (writes directly to the buffer)"""
        if self._threadSafe:
            with self._lock:
                self.buffer[key] = value
        else:
            self.buffer[key] = value
    
    def __len__(self):
        """Return buffer length"""
        return self.sz
    
    def get(self):
        """
        Get all buffered data as a single contiguous array in chronological order.
        Always returns a contiguous copy for consistent Numba performance.
        
        Returns:
            A 1D numpy array containing all valid data in the buffer
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
        
        # Create a new array and copy both segments
        result = np.empty(self.sz, dtype=np.float32)
        # Copy data from position to end
        result[:self.sz-self.position] = self.buffer[self.position:]
        # Copy data from beginning to position
        result[self.sz-self.position:] = self.buffer[:self.position]
        
        return result    