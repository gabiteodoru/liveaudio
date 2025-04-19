from liveaudio.PitchShiftVocoder import PitchShiftVocoder
from liveaudio.LinearVolumeNormalizer import LinearVolumeNormalizer
from liveaudio.utils import get_interactive_output_device, formatTimit
import numpy as np
import librosa
import matplotlib.pyplot as plt
import sounddevice as sd
import time

# Parameters
sr = 44100  # Sample rate
frSz = 4096  # Frame size
hop = 1024    # Hop size
duration = 2.0  # Duration in seconds
f0 = 440.0  # Frequency of the sine wave
output_device, sample_rate, input_channels = get_interactive_output_device()
play = lambda x: sd.play(x, sr, device=output_device)
def plot(x):
    plt.plot(x);plt.show()

# Generate sine wave
t = np.arange(0, duration, 1.0/sr)
input_signal = np.sin(2 * np.pi * f0 * t).astype(np.float64)
# The first frame will be all zeros, but that can be initialized like that when running realtime as well
frames = librosa.util.frame(np.concatenate((np.zeros(hop),input_signal)), frame_length=frSz+hop, hop_length=hop).T

st = 2**(1/12)
ratios = st**np.concatenate((np.linspace(0, -3, 30), np.linspace(-3, 3, 60)))
audioOut = np.array([])
voc = PitchShiftVocoder(sr, frSz, hop); voc.step(frames[1], .9); voc.step(frames[2], 1.1); # warmup
voc = PitchShiftVocoder(sr, frSz, hop)
lvn = LinearVolumeNormalizer()
for r, f in zip(ratios, frames):
    t0 = time.perf_counter()
    audio = lvn.normalize(voc.step(f, r))
    print(formatTimit(time.perf_counter()-t0))
    audioOut = np.concatenate((audioOut,audio))
plt.plot(audioOut);plt.show()
play(audioOut)