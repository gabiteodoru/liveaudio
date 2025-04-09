from scipy import fft
import numpy as np
from typing import Optional, Tuple
import numba
import librosa
from librosa.core.pitch import __check_yin_params as _check_yin_params
import scipy.stats
# from .realtimeHmm import onlineViterbiState, onlineViterbiStateOpt, blockViterbiStateOpt, sumProductViterbi
from .realtimeHmm import onlineViterbiState, blockViterbiStateOpt

def normalTransitionRow(k, v = None):
    if k<=0: return np.array([1.])
    if v is None: v = (k*(k+2))/6
    p = scipy.stats.norm.pdf(np.arange(k+1), 0, np.sqrt(v))
    return np.concatenate((p[:0:-1],p))


def autocorrelate1D(
    y: np.ndarray, max_size: Optional[int] = None) -> np.ndarray:
    """Bounded-lag auto-correlation

    Parameters
    ----------
    y : np.ndarray
        real array to autocorrelate
        len(y) should be power of 2
        
    max_size : int > 0 or None
        maximum correlation lag.

    Returns
    -------
    z : np.ndarray
        truncated autocorrelation ``y*y`` along the specified axis.[:max_size]

    Examples
    --------
    Compute full autocorrelation of ``y``

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> librosa.autocorrelate(y)
    array([ 6.899e+02,  6.236e+02, ...,  3.710e-08, -1.796e-08])

    Compute onset strength auto-correlation up to 4 seconds

    >>> import matplotlib.pyplot as plt
    >>> odf = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    >>> ac = librosa.autocorrelate(odf, max_size=4 * sr // 512)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(ac)
    >>> ax.set(title='Auto-correlation', xlabel='Lag (frames)')
    """
    n_pad = 2*len(y)
    # Compute the power spectrum along the chosen axis
    powspec = librosa.util.utils._cabs2(fft.rfft(y, n=n_pad))

    # Convert back to time domain
    autocorr = fft.irfft(powspec, n=n_pad)

    return autocorr[:max_size]

@numba.jit(nopython=True, cache=True)
def _realtime_cumulative_mean_normalized_difference(
    y_frame: np.ndarray,
    acf_frame: np.ndarray,
    min_period: int,
    max_period: int,
    tiny: float, # np.finfo(yin_denominator.dtype).tiny()
) -> np.ndarray:
    """Cumulative mean normalized difference function for a single frame
    
    Parameters
    ----------
    y_frame : np.ndarray [shape=(frame_length,)]
        audio time series for a single frame
    acf_frame : np.ndarray [shape=(max_period+1,)]
        pre-computed autocorrelation of y_frame up to max_period+1        
    min_period : int > 0 [scalar]
        minimum period
    max_period : int > 0 [scalar]
        maximum period
        
    Returns
    -------
    yin_frame : np.ndarray [shape=(max_period-min_period+1,)]
        Cumulative mean normalized difference function for the frame
    """
    # Prepare arrays for a single frame
    # Energy terms.
    yin_frame = np.cumsum(np.square(y_frame[:max_period+1]))

    # Difference function: d(k) = 2 * (ACF(0) - ACF(k)) - sum_{m=0}^{k-1} y(m)^2
    yin_frame[0] = 0
    yin_frame[1:] = (
        2 * (acf_frame[0:1] - acf_frame[1:]) - yin_frame[:max_period]
    )

    # Cumulative mean normalized difference function.
    yin_numerator = yin_frame[min_period : max_period + 1]
    # broadcast this shape to have leading ones

    cumulative_mean = (
        np.cumsum(yin_frame[1:]) / np.arange(1,max_period+1)
    )
    yin_denominator = cumulative_mean[min_period - 1 : max_period]
    yin_frame: np.ndarray = yin_numerator / (
        yin_denominator + tiny
    )
    return yin_frame

def cmnd(y_frame, min_period, max_period, tiny):
    acf_frame = autocorrelate1D(y_frame, max_period+1)
    return _realtime_cumulative_mean_normalized_difference(y_frame, acf_frame, min_period, max_period, tiny)


@numba.jit(nopython=True, cache=True)
def parabolicInterpolation(x: np.ndarray) -> np.ndarray:
    """Piecewise parabolic interpolation for yin and pyin on a single frame.

    Parameters
    ----------
    x : np.ndarray
        1D array to interpolate

    Returns
    -------
    parabolic_shifts : np.ndarray [shape=x.shape]
        position of the parabola optima (relative to bin indices)

        Note: the shift at bin `n` is determined as 0 if the estimated
        optimum is outside the range `[n-1, n+1]`.
    """
    # Allocate the output array
    shifts = np.empty_like(x)
    
    # Call the vectorized stencil on this single frame
    librosa.core.pitch._pi_wrapper(x, shifts)
    
    # Handle the edge condition not covered by the stencil
    shifts[0] = 0
    shifts[-1] = 0
    
    return shifts

@numba.guvectorize([
    (numba.float32[:], numba.float32[:], numba.boolean[:,:]),
    (numba.float64[:], numba.float64[:], numba.boolean[:,:]),
    (numba.float32[:], numba.float64[:], numba.boolean[:,:]),
    (numba.float64[:], numba.float32[:], numba.boolean[:,:])
], '(n),(m)->(n,m)', nopython=True, cache=True)
def numbaLessOuter(heights, thresh, result):
    for i in range(heights.shape[0]):
        for j in range(thresh.shape[0]):
            result[i, j] = heights[i] < thresh[j]

@numba.njit(cache=True)
def numbaCumsum(arr, axis=0):
    result = np.zeros_like(arr)
    if axis == 0:
        for i in range(arr.shape[0]):
            if i == 0:
                result[i] = arr[i]
            else:
                result[i] = result[i-1] + arr[i]
    elif axis == 1:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if j == 0:
                    result[i, j] = arr[i, j]
                else:
                    result[i, j] = result[i, j-1] + arr[i, j]
    return result

@numba.njit(cache=True)
def numbaCountNonzero(arr, axis=0):
    if axis == 0:
        result = np.zeros(arr.shape[1], dtype=np.int64)
        for j in range(arr.shape[1]):
            count = 0
            for i in range(arr.shape[0]):
                if arr[i, j]:
                    count += 1
            result[j] = count
        return result
    elif axis == 1:
        result = np.zeros(arr.shape[0], dtype=np.int64)
        for i in range(arr.shape[0]):
            count = 0
            for j in range(arr.shape[1]):
                if arr[i, j]:
                    count += 1
            result[i] = count
        return result

@numba.njit(cache=True)
def boltzmannPmf(k, lambda_param, N):
    # Boltzmann PMF implementation
    # P(k) = (1/Z) * exp(-lambda*k) for k = 0,1,...,N-1
    # where Z is the normalization constant = sum(exp(-lambda*i)) for i = 0,1,...,N-1
    
    # Calculate normalization constant
    Z = 0.0
    for i in range(N):
        Z += np.exp(-lambda_param * i)
    
    # Calculate PMF values
    result = np.zeros_like(k, dtype=np.float64)
    for i in range(len(k)):
        if 0 <= k[i] < N:
            result[i] = np.exp(-lambda_param * k[i]) / Z
    
    return result

@numba.njit(cache=True)
def pyin_single_frame(
    yin_frame,
    parabolic_shift,
    sr,
    thresholds,
    boltzmann_parameter,
    beta_probs,
    no_trough_prob,
    min_period,
    fmin,
    n_pitch_bins,
    n_bins_per_semitone,
):
    """
    Process a single frame with the PYIN algorithm.
    
    Parameters
    ----------
    yin_frame : np.ndarray
        Single YIN frame
    parabolic_shift : np.ndarray
        Parabolic interpolation shifts for this frame
    sr : int
        Sample rate
    thresholds : np.ndarray
        Array of thresholds for YIN algorithm
    boltzmann_parameter : float
        Boltzmann distribution parameter
    beta_probs : np.ndarray
        Beta distribution probabilities
    no_trough_prob : float
        Probability to assign when no trough is found
    min_period : int
        Minimum period in samples
    fmin : float
        Minimum frequency in Hz
    n_pitch_bins : int
        Number of pitch bins
    n_bins_per_semitone : int
        Number of bins per semitone
        
    Returns
    -------
    observation_probs : np.ndarray
        Observation probabilities for all pitch bins
    voiced_prob : float
        Probability that this frame is voiced
    """
    # Initialize defaults for empty case
    observation_probs = np.zeros(2 * n_pitch_bins)
    voiced_prob = 0
    
    # 2. Find the troughs
    is_trough = np.empty_like(yin_frame, dtype=np.bool_)  # Pre-allocate output array
    librosa.util.utils._localmin(yin_frame, is_trough)
    is_trough[-1] = yin_frame[-1] < yin_frame[-2]
    is_trough[0] = yin_frame[0] < yin_frame[1]
    (trough_index,) = np.nonzero(is_trough)

    yin_probs = np.zeros_like(yin_frame)
    
    if len(trough_index) > 0:
        # 3. Find troughs below each threshold
        trough_heights = yin_frame[trough_index]
        trough_thresholds = np.zeros((len(trough_heights), len(thresholds[1:])), dtype=np.bool_)
        numbaLessOuter(trough_heights, thresholds[1:], trough_thresholds)
        # 4. Define prior over troughs (smaller periods weighted more)
        trough_positions = numbaCumsum(trough_thresholds.astype(np.int32), axis=0) - 1
        n_troughs = numbaCountNonzero(trough_thresholds, axis=0)
        
        trough_prior = np.zeros_like(trough_positions, dtype=np.float64)
        for col in range(trough_positions.shape[1]):
            col_positions = trough_positions[:, col]
            n_value = int(n_troughs[col])
            col_result = boltzmannPmf(col_positions, boltzmann_parameter, n_value)
            for row in range(trough_positions.shape[0]):
                if trough_thresholds[row, col]:  # Only set value if threshold is True
                    trough_prior[row, col] = col_result[row]

        # 5. Calculate probabilities
        probs = np.zeros(trough_prior.shape[0])
        for i in range(trough_prior.shape[0]):
            for j in range(beta_probs.shape[0]):
                probs[i] += trough_prior[i, j] * beta_probs[j]
        
        global_min = np.argmin(trough_heights)
        n_thresholds_below_min = 0
        for j in range(trough_thresholds.shape[1]):
            if not trough_thresholds[global_min, j]:
                n_thresholds_below_min += 1

        probs[global_min] += no_trough_prob * np.sum(
            beta_probs[:n_thresholds_below_min]
        )
        
        yin_probs[trough_index] = probs
        
        # Get non-zero probabilities
        yin_period = []
        for i in range(len(yin_probs)):
            if yin_probs[i] > 0:
                yin_period.append(i)
        yin_period = np.array(yin_period)
        
        if len(yin_period) > 0:
            # Calculate period candidates
            period_candidates = np.zeros(len(yin_period))
            for i in range(len(yin_period)):
                period_candidates[i] = min_period + yin_period[i] + parabolic_shift[yin_period[i]]
            
            # Calculate f0 candidates
            f0_candidates = np.zeros(len(period_candidates))
            for i in range(len(period_candidates)):
                f0_candidates[i] = sr / period_candidates[i]
            
            # Calculate bin indices
            bin_index = np.zeros(len(f0_candidates), dtype=np.int64)
            for i in range(len(f0_candidates)):
                temp = 12 * n_bins_per_semitone * np.log2(f0_candidates[i] / fmin)
                temp = np.round(temp)
                if temp < 0:
                    bin_index[i] = 0
                elif temp >= n_pitch_bins:
                    bin_index[i] = n_pitch_bins - 1
                else:
                    bin_index[i] = int(temp)
            
            # Create observation probabilities
            observation_probs = np.zeros(2 * n_pitch_bins)
            
            # Map YIN probabilities to pitch bins
            for i in range(len(bin_index)):
                bin_idx = bin_index[i]
                observation_probs[bin_idx] += yin_probs[yin_period[i]]
            
            # Calculate voiced probability
            voiced_prob = 0.0
            for i in range(n_pitch_bins):
                voiced_prob += observation_probs[i]
            if voiced_prob > 1.0:
                voiced_prob = 1.0
    # Set unvoiced probabilities (happens in all cases)
    observation_probs[n_pitch_bins:] = (1 - voiced_prob) / n_pitch_bins
    
    return observation_probs, voiced_prob

class LivePyin:
    def __init__(self, fmin, fmax, sr=22050, frameLength=2048,  
                 hopLength=None, nThresholds=100, betaParameters=(2, 18),
                 boltzmannParameter=2, resolution=0.1, maxTransitionRate=35.92,
                 switchProb=0.01, noTroughProb=0.01, fillNa=np.nan,
                 viterbiMode = 'vanilla', dtype = np.float64,
                 nBinsPerVoicedSemitone = None, 
                 nBinsPerUnvoicedSemitone = None, 
                 maxSemitonesPerFrame = None,
                 transitionSemitonesVariance = None):
        
        # Store parameters
        self.dtype = dtype
        self.viterbiMode = viterbiMode
        self.tiny = np.finfo(dtype).tiny
        self.fmin = fmin
        self.fmax = fmax
        self.sr = sr
        self.frameLength = frameLength
        self.hopLength = frameLength // 4 if hopLength is None else hopLength
        self.fillNa = fillNa
        
        # Check parameters validity
        if fmin is None or fmax is None:
            raise ValueError('both "fmin" and "fmax" must be provided')
        
        _check_yin_params(
            sr=self.sr, 
            fmax=self.fmax, 
            fmin=self.fmin, 
            frame_length=self.frameLength, 
        )
        
        # Calculate minimum and maximum periods
        self.minPeriod = int(np.floor(sr / fmax))
        self.maxPeriod = min(int(np.ceil(sr / fmin)), frameLength - 1)
        
        # Initialize beta distribution for thresholds
        self.nThresholds = nThresholds
        self.betaParameters = betaParameters
        self.thresholds = np.linspace(0, 1, nThresholds + 1)
        betaCdf = scipy.stats.beta.cdf(self.thresholds, betaParameters[0], betaParameters[1])
        self.betaProbs = np.diff(betaCdf)
        
        # Initialize pitch bins
        self.resolution = resolution
        self.nBinsPerSemitone = int(np.ceil(1.0 / resolution))
        self.nPitchBins = int(np.floor(12 * self.nBinsPerSemitone * np.log2(fmax / fmin))) + 1
        
        # Boltzmann parameter for trough weighting
        self.boltzmannParameter = boltzmannParameter
        self.noTroughProb = noTroughProb
        
        # Initialize transition parameters
        self.maxTransitionRate = maxTransitionRate
        self.switchProb = switchProb
        
        # Compute transition matrix (which can be pre-computed)
        if nBinsPerVoicedSemitone is None:
            maxSemitonesPerFrame = round(maxTransitionRate * 12 * self.hopLength / sr)
            transitionWidth = maxSemitonesPerFrame * self.nBinsPerSemitone + 1
            
            # Construct the within voicing transition probabilities
            transition = librosa.sequence.transition_local(
                self.nPitchBins, transitionWidth, window="triangle", wrap=False
            )
            
            # Include across voicing transition probabilities
            tSwitch = librosa.sequence.transition_loop(2, 1 - switchProb)
            self.log_trans = np.log(np.kron(tSwitch, transition)+self.tiny)
        else:
            assert nBinsPerUnvoicedSemitone is not None
            assert maxSemitonesPerFrame is not None
            k = nBinsPerVoicedSemitone * maxSemitonesPerFrame
            v = transitionSemitonesVariance if transitionSemitonesVariance is not None else k*(k+2) / 6 # original variance from triangular window
            log_trans_0 = np.log(self.tiny+normalTransitionRow(k, v))
            f = (nBinsPerUnvoicedSemitone / nBinsPerVoicedSemitone)
            log_trans_1 = normalTransitionRow(int(np.ceil(k*f-0.5)), v*f*f)
            q = np.r_[:2*k+1]*f
            rq = np.ceil(q-0.5).astype(int)
            a=np.unique(rq, return_index=True)[1]
            rrq = np.ceil((a[:-1]+a[1:])/2-0.5).astype(int)
            self.correspondence = np.concatenate((rq, rrq))
            self.log_trans_00 = np.log(1-switchProb) + log_trans_0
            self.log_trans_01 = np.log(switchProb) + log_trans_0
            self.log_trans_10 = np.log(switchProb) + log_trans_1
            self.log_trans_11 = np.log(1-switchProb) + log_trans_1
            self.viterbiMode = 'block'
            self.n1 = k
        # Initialize probability state
        self.pInit = np.ones(2 * self.nPitchBins) / (2 * self.nPitchBins)
        self.hmmValue = np.log(self.pInit + self.tiny)

        
        # Pre-compute frequencies for each pitch bin
        self.freqs = fmin * 2 ** (np.arange(self.nPitchBins) / (12 * self.nBinsPerSemitone))
        
        # Initialize state for streaming
        self.currentState = None
        self.buffer = None
        # breakpoint()
        
    def step(self, y):
        yin_frame = cmnd(y, self.minPeriod, self.maxPeriod, self.tiny)
        parabolic_shift = parabolicInterpolation(yin_frame)
        observation_probs, voiced_prob = pyin_single_frame(
            yin_frame,
            parabolic_shift,
            self.sr,
            self.thresholds,
            self.boltzmannParameter,
            self.betaProbs,
            self.noTroughProb,
            self.minPeriod,
            self.fmin,
            self.nPitchBins,
            self.nBinsPerSemitone,
        )
        self.observation_probs = observation_probs
        # bestFreq = np.argmax(self.observation_probs)
        # print(max(self.observation_probs), self.freqs[bestFreq%self.nPitchBins], bestFreq<self.nPitchBins)
        # breakpoint()
        # self.hmmValue, state = onlineViterbiState(self.hmmValue, np.log(observation_probs+self.tiny), self.log_trans[200,:401])
        if self.viterbiMode == 'block':
            self.hmmValue, state = blockViterbiStateOpt(self.hmmValue, np.log(observation_probs+self.tiny), 
                self.log_trans_00, self.log_trans_01, self.log_trans_10, self.log_trans_11,
                self.n1, self.correspondence)
        else:            
            self.hmmValue, state = onlineViterbiState(self.hmmValue, np.log(observation_probs+self.tiny), self.log_trans)
        # breakpoint()
        # Find f0 corresponding to each decoded pitch bin.
        f0 = self.freqs[state % self.nPitchBins]
        voiced_flag = state < self.nPitchBins

        if not voiced_flag and self.fillNa is not None:
            f0 = self.fillNa

        return f0, voiced_flag, voiced_prob

def runRealtimePyinAsBatch(
    y: np.ndarray,
    *,
    fmin: float,
    fmax: float,
    sr: float = 22050,
    frame_length: int = 2048,
    hop_length: Optional[int] = None,
    n_thresholds: int = 100,
    beta_parameters: Tuple[float, float] = (2, 18),
    boltzmann_parameter: float = 2,
    resolution: float = 0.1,
    max_transition_rate: float = 35.92,
    switch_prob: float = 0.01,
    no_trough_prob: float = 0.01,
    fill_na: Optional[float] = np.nan,
    center: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert center is False
    lpyin = LivePyin(fmin, fmax, sr=22050, frameLength=2048,  
                 hopLength=None, nThresholds=100, betaParameters=(2, 18),
                 boltzmannParameter=2, resolution=0.1, maxTransitionRate=35.92,
                 switchProb=0.01, noTroughProb=0.01, fillNa=np.nan, 
                 dtype = y.dtype,
                 nBinsPerVoicedSemitone = 20, 
                 nBinsPerUnvoicedSemitone = 1, 
                 maxSemitonesPerFrame = 12,
                 transitionSemitonesVariance = None)
    if hop_length is None: hop_length = frame_length // 4 # Set the default hop if it is not already specified.
    y_frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length).T
    f0s, voiced_flags, voiced_probs = [],[],[]
    for yframe in y_frames:
        f0, voiced_flag, voiced_prob = lpyin.step(yframe)
        f0s.append(f0); voiced_flags.append(voiced_flag); voiced_probs.append(voiced_prob)
    return f0s, voiced_flags, voiced_probs