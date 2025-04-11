# LiveAudio

A real-time audio processing library built to provide efficient, streaming implementations of popular audio analysis algorithms, including real-time pitch tracking optimized for live streaming.

## Overview

LiveAudio is a Python package designed for real-time audio signal processing applications. It offers optimized, streaming implementations of audio algorithms that traditionally require full audio files, making them suitable for live audio processing. The library is built with performance in mind, using Numba for JIT compilation and optimized algorithms suitable for real-time applications.

Currently, LiveAudio implements:

- **LivePyin**: A real-time implementation of the probabilistic YIN (pYIN) algorithm for pitch detection
- **Real-time HMM**: Optimized Viterbi algorithm implementations for Hidden Markov Models in streaming contexts
- **CircularBuffer**: A high-performance circular buffer for managing audio frames in real-time

## Installation

# LiveAudio

A real-time implementation of librosa pyin for live audio streaming.

## Installation

You can install LiveAudio directly from PyPI:

```bash
pip install liveaudio
```

Alternatively, you can install from the source:

```bash
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
    frame_length=2048,  # Frame size
    hop_length=512   # Hop size
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

# Once the buffer is full (in this case, after 16 pushes), get the most recent frames
if buffer.full:
	frame = buffer.get()  # returns the full frame
```

## Usage Examples

### Basic Pitch Tracking

## Usage

Here's a basic example showing how to use LiveAudio for real-time pitch detection:

```python
from liveaudio.realtimePyin import LivePyin
from liveaudio.CircularBuffer import CircularBuffer
from liveaudio.utils import formatTimit, get_interactive_input_device
import time, librosa, numpy as np, sounddevice as sd

input_device, sample_rate, input_channels = get_interactive_input_device()
fmin, fmax = librosa.note_to_hz('C2'), librosa.note_to_hz('C5')
frame_size, hop_size = 4096, 1024
dtype = np.float64 # I'm not seeing a speed hit by using 64-bit when audio stream input is 32-bit (you can also tell the stream the type you'd like, but it doesn't support 64bit)

t0 = time.perf_counter()
lpyin = LivePyin(fmin, fmax, sr=sample_rate, frame_length=frame_size,  
             hop_length=hop_size, dtype=dtype,
             n_bins_per_semitone=20, max_semitones_per_frame=12,
             )
print('Class instantiated and code compiled in:', formatTimit(time.perf_counter()-t0))
print('Callback needs to run in: ',formatTimit(hop_size/sample_rate))

def audioCallback(indata, frames, timeInput, status):
    t0 = time.perf_counter()
    if status: print(status)
    x = indata if indata.ndim==1 else indata.mean(1)
    cb.push(x)
    if cb.full:
        y = cb.get()
        f0,voiced_flag,voiced_prob = lpyin.step(y.astype(dtype))
        print(f'Amplitude: {np.std(y):.3f}, {f0:.2f}Hz, {voiced_flag=}, {voiced_prob=:.2f}, cpu_time: ',formatTimit(time.perf_counter() - t0))
        
cb = CircularBuffer(frame_size, hop_size) 
for i in range(3): lpyin.warmup_and_reset() # warmup
time.sleep(0.1)

try:
    with sd.InputStream(device=input_device,
                  samplerate=sample_rate,
                  blocksize=hop_size,
                  channels=input_channels, 
                  callback=audioCallback,
                  latency = .015,
                  ):
        print("Audio routing started. Start singing now! Press Ctrl+C to stop")
        print("* Please remember that it won't work if your mic is off! (look at amplitude!, press mic-off key)")
        while True: time.sleep(0.1) # Keep the stream running
except KeyboardInterrupt:
    print("* Stopped by user")
except Exception as e:
    print(f"Error: {e}")
```

A [full working example](https://github.com/gabiteodoru/liveaudio/blob/main/demo.py) with device selection and more options is available in the [repository](https://github.com/gabiteodoru/liveaudio).

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