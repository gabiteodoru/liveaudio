import time

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
    t0, n = time.time(), 0
    while (t1:=time.time())-t0 < 1:
        r = f()
        n += 1
    sett((t1-t0)/n)
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