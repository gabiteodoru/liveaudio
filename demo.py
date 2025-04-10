from liveaudio.realtimePyin import LivePyin
from liveaudio.CircularBuffer import CircularBuffer
import sounddevice as sd
import time
import librosa
from liveaudio.timit import formatTimit
import numpy as np


# Find the devices by name
devices = sd.query_devices()
hostapis = sd.query_hostapis()

for i, d in enumerate(devices):
    if d['max_input_channels']>0:
        print(f"""{i}: {d['name']}; HostAPI: {hostapis[d['hostapi']]['name']};
    Sample Rate: {int(d['default_samplerate'])}; Channels: {d['max_input_channels']}\n""")
input_device = int(input('\nSelect audio input device: '))

sample_rate = int(devices[input_device]['default_samplerate'])
input_channels = int(devices[input_device]['max_input_channels'])

fmin, fmax = librosa.note_to_hz('C4'), librosa.note_to_hz('C5')
frame_size, hop_size = 4096, 1024

dtype = np.dtype(sd.default.dtype[0]).type # sounddevice streaming data default bit depth (e.g. np.float32)
# the LivePyin class needs to know the type
# you can also tell the stream the type you'd like
dtype = np.float64 # I'm not seeing a speed hit by using 64-bit when audio stream input is 32-bit

t0 = time.perf_counter()
lpyin = LivePyin(fmin, fmax, sr=sample_rate, frame_length=frame_size,  
             hop_length=hop_size, n_thresholds=100, beta_parameters=(2, 18),
             boltzmann_parameter=2, resolution=0.1, max_transition_rate=35.92,
             switch_prob=0.01, no_trough_prob=0.01, fill_na=np.nan, 
             dtype=dtype,
             n_bins_per_voiced_semitone=12, 
             n_bins_per_unvoiced_semitone=12, 
             max_semitones_per_frame=12,
             transition_semitones_variance=None
             )
print('Class instantiated and code compiled in:', formatTimit(time.perf_counter()-t0))

print('Callback needs to run in: ',formatTimit(hop_size/sample_rate))

def audioCallback(indata, frames, timeInput, status):
    if status:
        print(status)
    x = indata if indata.ndim==1 else indata.mean(1)
    cb.push(x)
    if cb.full:
        y = cb.get()
        f0,voiced_flag,voiced_prob = lpyin.step(y.astype(dtype))
        print(f'Amplitude: {np.std(y):.3f}, {f0:.2f}Hz, {voiced_flag=}, {voiced_prob=:.2f}')
        
# Start the stream
time.sleep(0.1)
cb = CircularBuffer(frame_size, hop_size)
for i in range(3):
    lpyin.warmupAndReset()
time.sleep(0.1)

try:
    with sd.InputStream(device=input_device,
                  samplerate=sample_rate,
                  blocksize=hop_size,
                  channels=input_channels, 
                  callback=audioCallback,
                  latency = .015,
                  ):
        
        print("* Audio routing started. Press Ctrl+C to stop")
        print("* Sing into your microphone to test")
        print("* Please remember that it won't work if your mic is off!")
        print("    (some computers have a hardware key; the demo prints amplitude, so it's a way to check)")
        
        # Keep the stream running
        while True:
            time.sleep(0.1)
except KeyboardInterrupt:
    print("* Stopped by user")
except Exception as e:
    print(f"Error: {e}")