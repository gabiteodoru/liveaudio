from liveaudio.buffers import OACBuffer
import numpy as np
import scipy.signal

# Create an OACBuffer (size must be a multiple of hop size)
frameSz, hopSz = 2048, 512
window = scipy.signal.windows.hann(frameSz) * (2/3) ** .5 # Energy preserving Hann window
oac_buffer = OACBuffer(
    size=frameSz,    # Frame size
    hop=hopSz,       # Hop size
    window=window    # Optional window function
)

# Create sample frames (in real usage, these would come from audio processing)
frames = [np.random.rand(frameSz) for _ in range(5)]

# Process frames through the OACBuffer
for frame in frames:
    output_hop = oac_buffer.pushGet(frame)  # Returns a hop-sized chunk
    # Each output_hop contains the overlapped sum of windowed frame segments
    print(f"Output hop shape: {output_hop.shape}")  # Should be (hopSz,)
