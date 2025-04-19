from liveaudio.LinearVolumeNormalizer import LinearVolumeNormalizer
import numpy as np
import matplotlib.pyplot as plt

# Parameters
sr = 44100  # Sample rate
duration = 0.5  # Duration in seconds
f0 = 440.0  # Frequency of the sine wave

# Generate sine wave with increasing amplitude from 0.5 to 2.0
t = np.arange(0, duration, 1.0/sr)
amplitude = np.linspace(0.5, 2.0, len(t))  # Increasing amplitude
audio = amplitude * np.sin(2 * np.pi * f0 * t)

# Create a normalizer and process audio
normalizer = LinearVolumeNormalizer(limit=0.95)
normalized_audio = normalizer.normalize(audio)

# Plot original and normalized audio
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.title("Original Audio (Amplitude: 0.5 to 2.0)")
plt.plot(t, audio)
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.title("Normalized Audio (Limit: 0.95)")
plt.plot(t, normalized_audio)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show() # plt.savefig('normalizer_demo.png', dpi=150)

# Print max amplitudes
print(f"Original audio max amplitude: {np.max(np.abs(audio)):.2f}")
print(f"Normalized audio max amplitude: {np.max(np.abs(normalized_audio)):.2f}")