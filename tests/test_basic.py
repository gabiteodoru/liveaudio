import librosa
import pytest
import numpy as np
import liveaudio.LivePyin as rp
from liveaudio.buffers import OACBuffer
from liveaudio.phase_vocoder_pitch_shift import getBlackmanHarrisWindow

@pytest.mark.parametrize("freq", [110, 220, 440, 880])
def test_pyin_tone_online(freq):
    y = librosa.tone(freq, duration=1.0)
    f0, _, _, = rp.run_realtime_pyin_as_batch(y, fmin=110, fmax=1000)
    assert np.allclose(np.log2(f0), np.log2(freq), rtol=0, atol=1e-2)

def test_oacbuffer_basic_functionality():
    # Test parameters
    frame_size = 1024
    hop_size = 256
    
    # Create buffer without a window to isolate buffer functionality
    buffer = OACBuffer(frame_size, hop_size)
    
    # Input a unit impulse (1.0 at position 0, zeros elsewhere)
    impulse = np.zeros(frame_size)
    impulse[0] = 1.0
    
    # Push the impulse into the buffer and collect multiple hops
    outputs = []
    outputs.append(buffer.pushGet(impulse))  # First hop
    
    # Push zeros to get the remaining parts of the impulse response
    for _ in range((frame_size // hop_size) - 1):
        outputs.append(buffer.pushGet(np.zeros(frame_size)))
    
    # Verify correct buffer behavior:
    # 1. Output length should be hop_size
    assert len(outputs[0]) == hop_size
    
    # 2. The impulse should propagate through consecutive hops
    # The first hop should have the impulse at position 0
    assert outputs[0][0] == 1.0
    
    # 3. Total energy conservation - the sum of all outputs should equal the input energy
    total_energy = sum(np.sum(output**2) for output in outputs)
    input_energy = np.sum(impulse**2)
    assert np.isclose(total_energy, input_energy), f"Energy not conserved: {total_energy} vs {input_energy}"
    
    # 4. Verify circular behavior by pushing another impulse
    buffer = OACBuffer(frame_size, hop_size)
    for i in range(frame_size // hop_size + 1):  # Go past one complete cycle
        test_impulse = np.zeros(frame_size)
        test_impulse[0] = 1.0 if i == 0 or i == frame_size // hop_size else 0.0
        output = buffer.pushGet(test_impulse)
        if i == frame_size // hop_size:  # Should match first output
            assert output[0] == 1.0, "Circular buffer behavior failed"

def test_BMWindow():
    sz = 4096
    hop = 1024
    r = sz//hop
    buffer = OACBuffer(sz, hop, getBlackmanHarrisWindow(sz, r))
    for i in range(r): buffer.pushGet(np.ones_like(buffer.window))
    x = buffer.buffer.sum(0)
    assert np.allclose(x.max(), 1)
    assert np.allclose(x.std(), 0)
