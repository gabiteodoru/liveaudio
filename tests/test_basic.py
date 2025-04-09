import librosa
import pytest
import numpy as np
import liveaudio.realtimePyin as rp

@pytest.mark.parametrize("freq", [110, 220, 440, 880])
def test_pyin_tone_online(freq):
    y = librosa.tone(freq, duration=1.0)
    f0, _, _, = rp.runRealtimePyinAsBatch(y, fmin=110, fmax=1000)
    assert np.allclose(np.log2(f0), np.log2(freq), rtol=0, atol=1e-2)
