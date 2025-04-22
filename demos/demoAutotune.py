from liveaudio.LivePyin import LivePyin
from liveaudio.buffers import CircularBuffer
from liveaudio.utils import formatTimit, get_interactive_input_device, get_interactive_output_device, findClosestNote, findClosestCMajorNote
from liveaudio.PitchShiftVocoder import PitchShiftVocoder
from liveaudio.LinearVolumeNormalizer import LinearVolumeNormalizer
import numpy as np
import librosa, time
import matplotlib.pyplot as plt
import sounddevice as sd

st = 2**(1/12)
input_device, sample_rate, input_channels = get_interactive_input_device()
output_device, _, output_channels = get_interactive_output_device()
play = lambda x: sd.play(x, sample_rate, device=output_device)
fmin, fmax = librosa.note_to_hz('C2'), librosa.note_to_hz('C6')
frame_length, hop = 4096, 1024
dtype = np.float64 # I'm not seeing a speed hit by using 64-bit when audio stream input is 32-bit (you can also tell the stream the type you'd like, but it doesn't support 64bit)

t0 = time.perf_counter()
lpyin = LivePyin(fmin, fmax, sr=sample_rate, frame_length=frame_length,  
             hop_length=hop, dtype=dtype,
             n_bins_per_semitone=20, max_semitones_per_frame=12,
             )
voc = PitchShiftVocoder(sample_rate, frame_length, hop, dtype = dtype)

print('Class instantiated and code compiled in:', formatTimit(time.perf_counter()-t0))
print('Callback needs to run in: ',formatTimit(hop/sample_rate))

lvn = LinearVolumeNormalizer()
cbPyin = CircularBuffer(frame_length, hop, dtype = dtype) 
cbVoc = CircularBuffer(frame_length+hop, hop, dtype = dtype); cbVoc.push(np.zeros(hop))

# Also save the input and output streams
inData, outData = [], []

def audioCallback(indata, outdata, frames, timeInput, status):
    t0 = time.perf_counter()
    if status: print(status)
    x = (indata if indata.ndim==1 else indata.mean(1)).astype(dtype)
    multiple = np.minimum(20,.95 / np.abs(x).max()) # my mic is a bit quiet, so this is a quick hack to up its volume
    x *= multiple # I could've used a separate LinearVolumeNormalizer object for the input as well
    inData.append(x)
    cbPyin.push(x)
    cbVoc.push(x)
    if cbPyin.full:
        y = cbPyin.get()
        f0,voiced_flag,voiced_prob = lpyin.step(y)
        if voiced_flag and np.isfinite(f0):
            r = findClosestCMajorNote(f0)/f0 # or findClosestNote
        else:
            r = 1.
        o = voc.step(cbVoc.get(), r) # you could use st instead of r to real-time pitch shift by some fixed amount
        result = lvn.normalize(o)
        print(f'Amplitude: {np.std(y):.3f}, {f0:.2f}Hz, {voiced_flag=}, {voiced_prob=:.2f}, cpu_time: ',formatTimit(time.perf_counter() - t0))
        outdata[:,0] = result
        outData.append(result)
    else: # until buffers fill, output the input
        outdata[:,0] = x
        outData.append(x)
for i in range(3): lpyin.warmup_and_reset() # warmup
time.sleep(0.1)

try:
    with sd.Stream(device=(input_device, output_device),
                  samplerate=sample_rate,
                  blocksize=hop,
                  channels=(input_channels, 1), # output in mono
                  callback=audioCallback,
                  latency = 15e-3, # 15ms
                  ):
        print("Audio routing started. Start singing now! Press Ctrl+C to stop")
        print("* Please remember that it won't work if your mic is off! (look at amplitude!, press mic-off key)")
        while True: time.sleep(0.1) # Keep the stream running
except KeyboardInterrupt:
    print("* Stopped by user")
except Exception as e:
    print(f"Error: {e}")
    
inData, outData = map(np.concatenate, (inData, outData))
plt.plot(inData); plt.show(); plt.plot(outData); plt.show()
# rng =range(80000,299999)
# play(inData[rng])
# play(outData[rng])
