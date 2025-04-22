import time, math
import sounddevice as sd

def findClosestNote(freqHz):
    '''Function to find the closest note to chromatic scale with A=440'''
    return 440.0 * 2**(math.floor(.5+math.log2(freqHz/440.0) * 12) / 12)

def findClosestCMajorNote(freqHz):
    '''
    [TODO] Test code:
    np.round(np.diff(np.log(np.unique(np.round([findClosestCMajorNote(i) for i in range(220,883)],2))))/np.log(st),2)
    Out[53]: array([2., 1., 2., 2., 1., 2., 2., 2., 1., 2., 2., 1., 2., 2.])
    '''
    semiTones = round(12 * math.log2(freqHz/440.0))
    cMajorOffsets = [0, 2, 4, 5, 7, 9, 11]
    position = semiTones % 12
    distances = [min((pos - position) % 12, (position - pos) % 12) for pos in cMajorOffsets]
    closestOffset = cMajorOffsets[distances.index(min(distances))]
    adjustment = closestOffset - position
    return freqHz * 2**(adjustment/12)

def get_interactive_input_device():
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
    
    return input_device, sample_rate, input_channels

def get_interactive_output_device():
    # Find the devices by name
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()

    for i, d in enumerate(devices):
        if d['max_output_channels']>0:
            print(f"""{i}: {d['name']}; HostAPI: {hostapis[d['hostapi']]['name']};
        Sample Rate: {int(d['default_samplerate'])}; Channels: {d['max_output_channels']}\n""")
    input_device = int(input('\nSelect audio output device: '))

    sample_rate = int(devices[input_device]['default_samplerate'])
    input_channels = int(devices[input_device]['max_output_channels'])
    
    return input_device, sample_rate, input_channels


_timit = None

def t(): 
    global _timit
    return _timit

def sett(x):
    global _timit
    _timit = x

def formatTimit(seconds):
    if seconds >= 1:
        return f"{seconds:.3f}s"
    elif seconds >= 1e-3:
        return f"{seconds*1e3:.3f}ms"
    elif seconds >= 1e-6:
        return f"{seconds*1e6:.3f}μs"
    else:
        return f"{seconds*1e9:.3f}ns"
    
def timit(f):
    t0, n = time.perf_counter(), 0
    while (t1:=time.perf_counter())-t0 < 1:
        r = f()
        n += 1
    sett((t1-t0)/n)
    print(formatTimit(_timit))
    return r

def tim(f):
    t0 = time.perf_counter()
    r = f()
    t1=time.perf_counter()
    sett((t1-t0))
    print(formatTimit(_timit))
    return r


def rtimit(f):
    t0, n = time.time(), 0
    while (t1:=time.time())-t0 < 1:
        r = f()
        n += 1
    sett((t1-t0)/n)
    return r, t()

def rstimit(f):
    t0, n = time.time(), 0
    while (t1:=time.time())-t0 < 1:
        r = f()
        n += 1
    sett((t1-t0)/n)
    return r, formatTimit(t())