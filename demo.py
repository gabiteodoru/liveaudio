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