# LiveAudio

A real-time audio processing library built to provide efficient, streaming implementations of popular audio analysis algorithms.

## Overview

LiveAudio is a Python package designed for real-time audio signal processing applications. It offers optimized, streaming implementations of audio algorithms that traditionally require full audio files, making them suitable for live audio processing. The library is built with performance in mind, using Numba for JIT compilation and optimized algorithms suitable for real-time applications.

Currently, LiveAudio implements:

- **LivePyin**: A real-time implementation of the probabilistic YIN (pYIN) algorithm for pitch detection
- **Real-time HMM**: Optimized Viterbi algorithm implementations for Hidden Markov Models in streaming contexts
- **CircularBuffer**: A high-performance circular buffer for managing audio frames in real-time

## Installation

```bash
# Not yet available on PyPI
# Clone the repository for now
git clone https://github.com/gabiteodoru/liveaudio.git
cd liveaudio
pip install -e .
```

## Dependencies

- numpy
- numba
- scipy
- librosa (>= 0.11.0)

## Components

### LivePyin

A real-time implementation of the probabilistic YIN (pYIN) algorithm for fundamental frequency (F0) estimation. Unlike the standard pYIN implementation in librosa, LivePyin is designed for frame-by-frame processing, making it suitable for real-time applications.

```python
from liveaudio import LivePyin

# Initialize the LivePyin instance
lpyin = LivePyin(
    fmin=65.0,      # Minimum frequency in Hz
    fmax=2093.0,    # Maximum frequency in Hz
    sr=44100,       # Sample rate
    frameLength=2048,  # Frame size
    hopLength=512   # Hop size
)

# Process frames one by one
for frame in audio_frames:
    f0, voiced_flag, voiced_prob = lpyin.step(frame)
    # f0: Estimated fundamental frequency
    # voiced_flag: Boolean indicating if the frame is voiced
    # voiced_prob: Probability of the frame being voiced
```

### Real-time HMM

Efficient implementations of the Viterbi algorithm optimized for real-time processing with Hidden Markov Models. The module provides several variants:

- `onlineViterbiState`: Basic online Viterbi algorithm
- `onlineViterbiStateOpt`: Optimized Viterbi for sparse transition matrices
- `blockViterbiStateOpt`: Block-structured Viterbi algorithm for specialized transition matrices
- `sumProductViterbi`: Sum-product variant of the Viterbi algorithm

These implementations are particularly useful for pitch tracking and other sequential estimation problems in audio processing.

### CircularBuffer

A high-performance circular buffer implementation for managing audio frames efficiently in real-time processing contexts.

```python
from liveaudio import CircularBuffer

# Create a circular buffer (size must be a multiple of hop size)
buffer = CircularBuffer(
    sz=8192,        # Total buffer size
    hopSize=512,    # Size of data chunks for updates
    threadSafe=True # Whether to use thread-safe operations
)

# Push new audio frames
buffer.push(new_audio_frame)  # Must be a 1D numpy array of size hopSize

# Get the most recent frames
frames = buffer.getFrames(numFrames=4)  # Returns a (numFrames, hopSize) array
```

## Usage Examples

### Basic Pitch Tracking

```python
import numpy as np
from liveaudio import LivePyin, CircularBuffer

# Initialize components
sr = 44100  # Sample rate
frame_size = 2048
hop_size = 512

pyin = LivePyin(fmin=65.0, fmax=1000.0, sr=sr, frameLength=frame_size, hopLength=hop_size)
buffer = CircularBuffer(sz=frame_size, hopSize=hop_size)

# In a real-time audio processing loop:
def process_audio_callback(new_audio_chunk):
    # Add new audio to the buffer
    buffer.push(new_audio_chunk)
    
    # Get the latest complete frame
    latest_frame = buffer.getFrames(numFrames=1)[0]
    
    # Estimate pitch
    f0, is_voiced, confidence = pyin.step(latest_frame)
    
    if is_voiced:
        print(f"Detected pitch: {f0:.1f} Hz (confidence: {confidence:.2f})")
    else:
        print("No pitch detected")
```

## Future Plans

- Live AutoTune algorithm
- Real-time spectral processing tools
- More audio effects processing
- Support for audio I/O through PyAudio or similar libraries

## License

MIT License

## Acknowledgments

This package builds upon concepts and algorithms from the [librosa](https://librosa.org/) library, providing real-time compatible implementations. Special thanks to the librosa team for their excellent work in audio signal processing.
This README was written by Claude.AI . 