import scipy.fft
import numpy as np
from liveaudio.buffers import OACBuffer
import librosa
from numpy import pi
import math
import scipy.signal

pi2 = 2*pi

def isPowerOf2(x):
    """Check if a number is a power of 2 (has exactly one bit set).
    Used for optimizing FFT operations which work best with power-of-2 sizes."""
    try:
        return x == int(x) and x > 0 and (int(x) & (int(x) - 1)) == 0
    except (TypeError, ValueError):
        return False

def isPowerOf2Multiple(larger, smaller):
    """Check if larger/smaller is a power of 2.
    Used to verify that FFT size is a power-of-2 multiple of frame size."""
    # Check if smaller divides larger evenly first
    if larger % smaller != 0:
        return False
    return isPowerOf2(larger // smaller)


class PitchShiftVocoder:
    """Phase vocoder implementation for high-quality pitch shifting of audio signals.
    
    Uses FFT/IFFT transforms and phase manipulation to shift pitch while preserving timing.
    Implements the classic phase vocoder algorithm with improvements for handling
    pitch increases that require time stretching.
    """
    def __init__(self, sr, frSz, hop=None, dtype=np.float64, fftSz=None):
        """Initialize the pitch shifter.
        
        Args:
            sr: Sample rate in Hz
            frSz: Frame size in samples (window size)
            hop: Hop size in samples (defaults to and is only implemented for frSz//4)
            dtype: Data type for numerical processing
            fftSz: FFT size (defaults to frame size, not yet implemented for other values)
        """
        if hop is None: hop = frSz//4
        assert hop*4 == frSz, "Only implemented for hop*4 == frame size due to window COLA property"
        assert all(isPowerOf2(i) for i in (hop, frSz)), "hop and frame size should be powers of 2 (technically not absolutely necessary, but no good reason I can't think of not to do it)"
        assert fftSz is None or fftSz == frSz, "Different FFT size not yet implemented"
        if fftSz is None:
            fftSz = frSz
        else:
            assert isPowerOf2Multiple(fftSz, frSz), "FFT size must be an exact power-of-2 multiple of frame size"
        self.fftSz = fftSz
        self.sr = sr
        self.frSz = frSz
        self.hop = hop
        self.r = frSz // hop  # Overlap ratio (typically 4)
        self.k = fftSz // 2 + 1  # Real FFT output size (number of frequency bins)
        self.window = scipy.signal.windows.hann(frSz) * (2/3)**.5  # Apply Hann window with 75% overlap-add normalization to keep energy constant
        self.buffer = OACBuffer(frSz, hop, self.window.copy())  # Overlap-add buffer for frame synthesis
        # Zero-pad buffer for FFT if needed
        self.fftbuf = np.zeros(self.fftSz, dtype=dtype) if self.fftSz > self.frSz else None
        self.dtype = dtype
        # State variables for tracking phase across frames
        self.phase_acc = np.zeros(self.k, dtype=dtype)  # Phase accumulator
        self.last_magnitude = np.zeros(self.k, dtype=dtype)  # Previous frame magnitude spectrum
        self.last_phase = np.zeros(self.k, dtype=dtype)  # Previous frame phase spectrum
        self.lastRatio = 1.  # Previous pitch ratio for frame timing adjustments; used with a different meaning if previous frame was pitch-up

    def _makeAudio(self, R):
        """Convert magnitude spectrum to time-domain audio using accumulated phase.
        
        Args:
            R: Magnitude spectrum
            
        Returns:
            Time-domain audio frame
        """
        X = librosa.util.phasor(self.phase_acc, mag=R)  # Convert magnitude and phase to complex spectrum
        x = scipy.fft.irfft(X)  # Inverse FFT to get time domain signal
        if self.fftbuf is not None: x = x[:self.frSz]  # Truncate if zero-padded FFT was used
        return x

    def _fft(self, audio):
        """Compute FFT of windowed audio frame.
        
        Args:
            audio: Input audio frame
            
        Returns:
            Tuple of (magnitude spectrum, phase spectrum)
        """
        if self.fftbuf is not None:
            # Zero-pad if FFT size > frame size
            self.fftbuf[:self.frSz] = audio*self.window
            spectrum = scipy.fft.rfft(self.fftbuf)
        else:
            spectrum = scipy.fft.rfft(self.window*audio)
        R = np.abs(spectrum)  # Magnitude spectrum
        phi = np.angle(spectrum)  # Phase spectrum
        return R, phi
        
    def step(self, audio, ratio):
        """Apply pitch shifting with phase vocoder algorithm.
        
        Args:
            audio: Input audio frame (must be of size hop + frSz)
              - When running in real-time, note that the first hop-worth of samples represent data from the past
              - That means, practically, that you should initialize the buffer with hop-worth of zeros, then the first frSz-worth of samples
              would represent the present
            ratio: Pitch shift ratio (>1 increases pitch, <1 decreases pitch)
              - Intended for use with minor pitch-shifts e.g. autotune
              - Not yet implemented for pitch-shifts > 1.25 (slightly less than a Major 3rd)
                  - Should you want to extend it, note that you'd probably need to modify the cross-fade synthesis
                  to use 3 or more frSz-windows rather than the hard-coded two
                  This will necessitate choosing appropriate windows, recomputing the limits, etc.
              - For smooth audio without artifacts, try to change ratio smoothly between consecutive frames
            
        Returns:
            Pitch-shifted audio frame of size hop
            
        Note:
            This implements a frame-by-frame phase vocoder with special handling for
            pitch increases that require time stretching.
        """
        assert ratio <= 1+1/self.r, "pitch shifting up by more than a factor of (1+frameSize/hop) not implemented"
        assert len(audio) == self.hop + self.frSz, "Input audio size doesn't match that passed in the constructor"
        warpedSz = 2*math.floor(self.frSz*ratio/2+.5)  # Always even sized for better windowing
        pitchUp = warpedSz > self.frSz
        # pitchDown = warpedSz < self.frSz
        noChange = warpedSz == self.frSz
        if noChange: ratio = 1. # if ratio is so close to 1. we can't even add or remove 2 samples, keep it unchanged
        R0, phi0 = self._fft(audio[self.hop:]) # Get magnitude and phase of current frame

        # Update phase accumulator based on previous frame
        if self.lastRatio == 1: # Standard phase update when previous frame had no pitch change
            self.phase_acc += phi0 - self.last_phase
        else: 
            # For variable pitch ratios, need to adjust phase advancement
            # We need to advance phase less than a whole frame, because previous frame
            # was shortened then slowed down; or it was sped up, but due to synthesis
            # we've already advanced the phase so much that less than a full hop is left to advance
            # So note that for previou pitch-up, it doesn't have the same semantics
            Rmr, phimr = self._fft(audio[int(self.hop*self.lastRatio):self.frSz+int(self.hop*self.lastRatio)])
            self.phase_acc += phimr - self.last_phase
            
        if pitchUp: 
            # For pitch up, we need to stretch the frame and crossfade two frames
            # Prepare empty array and windows for crossfade
            x = np.zeros_like(audio, shape=(warpedSz,))
            endSz = warpedSz - self.frSz  # Always even - amount of stretching needed
            wEndSz = 2 * endSz  # Multiply by 2 to avoid numerical errors in dividing Hann windows
            wOverlapSz = warpedSz-2 * wEndSz  # Overlap region for crossfade
            # Create new window and crossfade weights
            newWin = scipy.signal.windows.hann(warpedSz) * (2/3)**.5
            crossWin = np.cos(np.linspace(0, pi/2, wOverlapSz))
            with np.errstate(divide='ignore', invalid='ignore'): # Handle division by zero in window normalization
                invWindowEdge = np.nan_to_num(newWin[:wEndSz]/self.window[:wEndSz], nan=((self.frSz-1)/(warpedSz-1))**2)  # Theoretical value of the limit
            invWindowOverlap = newWin[wEndSz:-wEndSz]/self.window[wEndSz:wOverlapSz+wEndSz]*crossWin**2
            w = np.concatenate((invWindowEdge, invWindowOverlap))
            # Synthesize audio from previous and current frame and crossfade
            Rmr, phimr = self._fft(audio[self.hop-endSz:self.frSz+self.hop-endSz])
            x1 = self._makeAudio(self.last_magnitude)
            x[:len(w)] = x1[:len(w)]*w  # Apply first part of crossfade (previous frame)
            self.phase_acc += phi0 - phimr
            x2 = self._makeAudio(R0)
            x[-len(w):] += x2[-len(w):]*w[::-1]  # Apply second part of crossfade (current frame)
            ratio -= endSz/self.hop  # Ratio will be the proportion of hop needed to be phase-accumulated
             # to account for a stretched frame given that we have already advanced by phi0 - phimr
             # in the middle of creating the stretched audio
        else: # For no change or pitch down, synthesis is simpler
            x = self._makeAudio(R0)[:warpedSz]  # Use current frame, chop off excess for pitch down
            
        if not noChange: # Resample to match target frame size if pitch was changed 
            x = librosa.resample(x, orig_sr=len(x), target_sr=self.frSz)
            
        # Store current frame data for next iteration
        self.last_magnitude, self.last_phase = R0, phi0
        self.lastRatio = ratio
        
        # Push to overlap-add buffer and get output frame
        return self.buffer.pushGet(x)