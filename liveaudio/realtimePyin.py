from scipy import fft
import numpy as np
from typing import Optional, Tuple, Union, Literal, Any
import numba
import librosa
from librosa.core.pitch import __check_yin_params as _check_yin_params
import scipy.stats
# from .realtimeHmm import onlineViterbiState, onlineViterbiStateOpt, blockViterbiStateOpt, sumProductViterbi
from .realtimeHmm import onlineViterbiState, blockViterbiStateOpt
# import warnings

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
    use_correspondence,
    correspondence,
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
    # Create observation probabilities
    lOutput = len(correspondence) if use_correspondence else 2 * n_pitch_bins
    observation_probs = np.zeros(lOutput, dtype=yin_frame.dtype)
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
        
        trough_prior = np.zeros_like(trough_positions, dtype=yin_frame.dtype)
        for col in range(trough_positions.shape[1]):
            col_positions = trough_positions[:, col]
            n_value = int(n_troughs[col])
            col_result = boltzmannPmf(col_positions, boltzmann_parameter, n_value)
            for row in range(trough_positions.shape[0]):
                if trough_thresholds[row, col]:  # Only set value if threshold is True
                    trough_prior[row, col] = col_result[row]

        # 5. Calculate probabilities
        probs = np.zeros(trough_prior.shape[0], dtype=yin_frame.dtype)
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
            period_candidates = np.zeros(len(yin_period), dtype=yin_frame.dtype)
            for i in range(len(yin_period)):
                period_candidates[i] = min_period + yin_period[i] + parabolic_shift[yin_period[i]]
            
            # Calculate f0 candidates
            f0_candidates = np.zeros(len(period_candidates), dtype=yin_frame.dtype)
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
    if use_correspondence:
        observation_probs[n_pitch_bins:] = (1 - voiced_prob) / n_pitch_bins
    else:
        l = len(bin_index)
        for i in range(len(bin_index)):
            observation_probs[l+correspondence[i]] += observation_probs[i]
    
    return observation_probs, voiced_prob

def findClosestIndex(i_values, delta_a, delta_b):
    numerator = i_values * delta_a
    quotient, remainder = np.divmod(numerator, delta_b)
    # If remainder/delta_b < 0.5, return floor, otherwise return ceiling
    # This is equivalent to checking if 2*remainder < delta_b
    result = np.where(2 * remainder < delta_b, quotient, quotient + 1)
    return result

# actually same as findClosestIndex, but with flipped delta_a, delta_b
# def findClosestIndexInverseStep(i_values, qa, qb):
#     # Calculate i*(qb/qa) = i*qb/qa
#     numerator = i_values * qb
#     quotient, remainder = np.divmod(numerator, qa)
#     # Compare 2*remainder vs qa to determine rounding
#     result = np.where(2 * remainder < qa, quotient, quotient + 1)
#     return result

# Viterbi modes
_VIT_VANILLA = 0 # same method as used by librosa.pyin, probably too slow for live
_VIT_T22 = 1 # take advantage of the sparse 2x2 Block Toeplitz structure of the transition matrix
_VIT_T22_NEQ = 2 # same as above, but allow different sizes for the blocks
_VIT_SPM = 3 # use sum-product for the past, max for the present (SPM); operates in probability rather than log-domain; vanilla
_VIT_SPMT22 = 4 # SPM, 2x2 Block Toeplitz structure of the transition matrix
_VIT_SPMT22_NEQ = 5 # same as above, but allow different sizes for the blocks

class LivePyin:
    def __init__(self, 
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
                 fill_na: Union[float, Any] = np.nan,
                 dtype: np.dtype = np.float64,
                 viterbi_mode: Literal['vanilla', 'fast'] = 'fast', 
                 n_bins_per_semitone: Optional[int] = None, 
                 n_bins_per_unvoiced_semitone: Optional[int] = None, 
                 max_semitones_per_frame: Optional[int] = None,
                 transition_distribution: Literal['triangular', 'normal'] = 'normal',
                 transition_semitones_variance: Optional[float] = None,
                 ):
        """Real-time fundamental frequency (F0) estimation using probabilistic YIN (pYIN).
    
        This is a streaming implementation of the pYIN algorithm for real-time audio processing.
        It is based on the librosa 0.11.0 implementation of pYIN but adapted for frame-by-frame
        processing in live audio applications.
    
        pYIN [#]_ is a modification of the YIN algorithm [#]_ for fundamental frequency (F0) estimation.
        In the first step of pYIN, F0 candidates and their probabilities are computed using the YIN algorithm.
        In the second step, Viterbi decoding is used to estimate the most likely F0 sequence and voicing flags.
    
        .. [#] Mauch, Matthias, and Simon Dixon.
            "pYIN: A fundamental frequency estimator using probabilistic threshold distributions."
            2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2014.
    
        .. [#] De Cheveigné, Alain, and Hideki Kawahara.
            "YIN, a fundamental frequency estimator for speech and music."
            The Journal of the Acoustical Society of America 111.4 (2002): 1917-1930.
    
        Parameters
        ----------
        fmin : number > 0 [scalar]
            Minimum frequency in Hertz.
            The recommended minimum is ``librosa.note_to_hz('C2')`` (~65 Hz)
            though lower values may be feasible.
    
        fmax : number > fmin, <= sr/2 [scalar]
            Maximum frequency in Hertz.
            The recommended maximum is ``librosa.note_to_hz('C7')`` (~2093 Hz)
            though higher values may be feasible.
    
        sr : number > 0 [scalar]
            Sampling rate in Hertz. Default: 22050.
    
        frame_length : int > 0 [scalar]
            Length of the frames in samples.
            By default, ``frame_length=2048`` corresponds to a time scale of about 93 ms at
            a sampling rate of 22050 Hz.
    
        hop_length : None or int > 0 [scalar]
            Number of audio samples between adjacent pYIN predictions.
            If ``None``, defaults to ``frame_length // 4``.
    
        n_thresholds : int > 0 [scalar]
            Number of thresholds for peak estimation. Default: 100.
    
        beta_parameters : tuple
            Shape parameters for the beta distribution prior over thresholds. Default: (2, 18).
    
        boltzmann_parameter : number > 0 [scalar]
            Shape parameter for the Boltzmann distribution prior over troughs.
            Larger values will assign more mass to smaller periods. Default: 2.
    
        resolution : float in (0, 1)
            Resolution of the pitch bins.
            0.01 corresponds to cents. Default: 0.1.
    
        max_transition_rate : float > 0
            Maximum pitch transition rate in octaves per second. Default: 35.92.
    
        switch_prob : float in (0, 1)
            Probability of switching from voiced to unvoiced or vice versa. Default: 0.01.
    
        no_trough_prob : float in (0, 1)
            Maximum probability to add to global minimum if no trough is below threshold. Default: 0.01.
    
        fill_na : float or ``np.nan``
            Default value for unvoiced frames of ``f0``. Default: np.nan.
    
        dtype : numpy.dtype
            Data type for internal calculations. Default: np.float64.
    
        viterbi_mode : str, {'vanilla', 'fast'}
            Mode for Viterbi algorithm implementation:
            - 'vanilla': Uses the original librosa implementation
            - 'fast': Takes advantage of the sparse 2x2 block Toeplitz structure of the transition matrix
              for faster processing with some trade-offs in accuracy for edge cases.
            Default: 'fast'
            This feature is experimental and additional modes may be added in future versions.
    
        n_bins_per_semitone : int or None
            Custom number of bins per semitone. If provided, this overrides the internal calculation
            based on `resolution`. Default: None (use librosa's internal computation).
    
        n_bins_per_unvoiced_semitone : int or None
            Ignored for viterbi_mode = 'vanilla' 
            Experimental parameter for setting a different (typically lower) bin resolution for
            unvoiced states to reduce computational complexity. When None, uses the same
            resolution as for voiced states. Default: None.
    
        max_semitones_per_frame : int or None
            Custom maximum semitone transition limit. If provided, this overrides the internal
            calculation based on `max_transition_rate`. Default: None.
    
        transition_distribution : str, {'triangular', 'normal'}
            Distribution type used for modeling pitch transitions:
            - 'triangular': Uses triangular distribution as in the original librosa implementation
            - 'normal': Uses normal distribution, which is more appropriate when using different
              resolutions for voiced and unvoiced states.
            Default: 'normal'.

        transition_semitones_variance : float or None
            Variance parameter when using normal distribution for transitions. Default: None.
            If unspecified, defaults to k*(k+2) / 6, where k is the number of non-zero transitions
            This is the variance of the original librosa.pyin triangular distribution
        
        Notes
        -----
        Compared to librosa's pYIN implementation, this streaming version:
        1. Removes `center` parameter (not applicable in streaming context)
        2. Removes `pad_mode` parameter (not applicable in streaming context)
        3. Removes `win_length` parameter, which is deprecated in librosa 0.11.0
        4. Adds streaming-specific parameters for performance optimization
    
        Returns
        -------
        When processing a frame, the method returns:
        f0: float
            Fundamental frequency in Hertz for the current frame.
        voiced_flag: bool
            Boolean flag indicating whether the current frame is voiced or not.
        voiced_prob: float
            Probability that the current frame is voiced.
    
        See Also
        --------
        librosa.pyin:
            Original batch implementation of the pYIN algorithm in librosa 0.11.0 (https://librosa.org).
        
        Examples
        --------
        For usage examples, please refer to the README.md in the liveaudio library
        documentation which demonstrates a complete real-time processing workflow.
        """        
        # Store parameters
        self.dtype = dtype
        self.viterbi_mode = viterbi_mode
        self.tiny = np.finfo(dtype).tiny
        self.fmin = dtype(fmin)
        self.fmax = dtype(fmax)
        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = frame_length // 4 if hop_length is None else hop_length
        self.fill_na = fill_na
        
        # Check parameters validity
        if fmin is None or fmax is None:
            raise ValueError('both "fmin" and "fmax" must be provided')
        
        _check_yin_params(
            sr=self.sr, 
            fmax=self.fmax, 
            fmin=self.fmin, 
            frame_length=self.frame_length, 
        )
        
        # Calculate minimum and maximum periods
        self.min_period = int(np.floor(sr / fmax))
        self.max_period = min(int(np.ceil(sr / fmin)), frame_length - 1)
        
        # Initialize beta distribution for thresholds
        self.n_thresholds = n_thresholds
        self.beta_parameters = beta_parameters
        self.thresholds = np.linspace(0, 1, n_thresholds + 1).astype(dtype)
        beta_cdf = scipy.stats.beta.cdf(self.thresholds, beta_parameters[0], beta_parameters[1]).astype(dtype)
        self.beta_probs = np.diff(beta_cdf)
        
        # Initialize pitch bins
        self.resolution = resolution
        if n_bins_per_semitone is None:
            n_bins_per_semitone = int(np.ceil(1.0 / resolution))
        self.n_pitch_bins = int(np.floor(12 * n_bins_per_semitone * np.log2(fmax / fmin))) + 1
        
        # Boltzmann parameter for trough weighting
        self.boltzmann_parameter = boltzmann_parameter
        self.no_trough_prob = no_trough_prob
        
        # Initialize transition parameters
        self.switch_prob = switch_prob
        
        self._viterbi_mode = _VIT_T22 if viterbi_mode == 'fast' else _VIT_VANILLA
        if max_semitones_per_frame is None:
            max_semitones_per_frame = round(max_transition_rate * 12 * self.hop_length / sr)

        transition_width = max_semitones_per_frame * n_bins_per_semitone + 1
    
        # Compute transition matrix (which can be pre-computed)
        if self._viterbi_mode == _VIT_VANILLA:
            
            # Construct the within voicing transition probabilities
            transition = librosa.sequence.transition_local(
                self.n_pitch_bins, transition_width, window="triangle", wrap=False
            )
            
            # Include across voicing transition probabilities
            t_switch = librosa.sequence.transition_loop(2, 1 - switch_prob)
            self.log_trans = np.log(np.kron(t_switch, transition)+self.tiny)
            # Pre-compute frequencies for each pitch bin
            self.freqs = fmin * 2 ** (np.arange(self.n_pitch_bins) / (12 * n_bins_per_semitone))
            # Initialize probability state
            self.p_init = np.ones(2 * self.n_pitch_bins) / (2 * self.n_pitch_bins)
            self.hmm_value = np.log(self.p_init + self.tiny)
        else:
            if n_bins_per_unvoiced_semitone is None:
                n_bins_per_unvoiced_semitone = n_bins_per_semitone
            if n_bins_per_semitone != n_bins_per_unvoiced_semitone:
                self._viterbi_mode == _VIT_T22_NEQ
                # The code below handles both equal and unequal cases, may refine in a later version
            # Pre-compute frequencies for each pitch bin
            f = (n_bins_per_unvoiced_semitone / n_bins_per_semitone)
            self.n_pitch_bins = self.n_pitch_bins1 = int(np.floor(12 * n_bins_per_semitone * np.log2(fmax / fmin))) + 1
            self.n_pitch_bins2 = int(np.floor(12 * n_bins_per_unvoiced_semitone * np.log2(fmax / fmin))) + 1
            range1, range2 = np.arange(self.n_pitch_bins1), np.arange(self.n_pitch_bins2)
            self.correspondence = np.concatenate((
                findClosestIndex(range1, n_bins_per_unvoiced_semitone, n_bins_per_semitone),
                findClosestIndex(range2, n_bins_per_semitone, n_bins_per_unvoiced_semitone),
                    ))
            self.freqs1 = fmin * 2 ** (range1 / (12 * n_bins_per_semitone))
            self.freqs2 = fmin * 2 ** (range2 / (12 * n_bins_per_unvoiced_semitone))
            self.n_pitch_bins1 = len(self.freqs1)
            k = n_bins_per_semitone * max_semitones_per_frame
            v = transition_semitones_variance if transition_semitones_variance is not None else k*(k+2) / 6 # original variance from triangular window
            log_trans_0 = np.log(self.tiny+normalTransitionRow(k, v))
            log_trans_1 = np.log(self.tiny+normalTransitionRow(int(np.ceil(k*f-0.5)), v*f*f))
            self.log_trans_00 = np.log(1-switch_prob) + log_trans_0
            self.log_trans_01 = np.log(switch_prob) + log_trans_0
            self.log_trans_10 = np.log(switch_prob) + log_trans_1
            self.log_trans_11 = np.log(1-switch_prob) + log_trans_1
            self.n1 = self.n_pitch_bins1
            # Initialize probability state
            self.p_init = np.concatenate((np.ones(self.n_pitch_bins1) / self.n_pitch_bins1, np.ones(self.n_pitch_bins2) / self.n_pitch_bins2))/2
            self.hmm_value = np.log(self.p_init + self.tiny)
        
        self.n_bins_per_semitone = n_bins_per_semitone
        # Initialize state for streaming
        self.current_state = None
        self.buffer = None
        self.warmup_and_reset()
        # breakpoint()
        
    def warmup_and_reset(self):
        # Force JIT compilation so it doesn't happen while streaming
        hmm_value = self.hmm_value
        self.step(np.random.randn(self.frame_length))
        self.hmm_value = hmm_value 
        
    def step(self, y):
        yin_frame = cmnd(y, self.min_period, self.max_period, self.tiny)
        parabolic_shift = parabolicInterpolation(yin_frame)
        fast_matrix_mode = self._viterbi_mode == _VIT_T22 or self._viterbi_mode == _VIT_T22_NEQ
        observation_probs, voiced_prob = pyin_single_frame(
            yin_frame,
            parabolic_shift,
            self.sr,
            self.thresholds,
            self.boltzmann_parameter,
            self.beta_probs,
            self.no_trough_prob,
            self.min_period,
            self.fmin,
            self.n_pitch_bins,
            self.n_bins_per_semitone,
            fast_matrix_mode,
            self.correspondence,
        )
        # self.observation_probs = observation_probs
        # bestFreq = np.argmax(self.observation_probs)
        # print(max(self.observation_probs), self.freqs[bestFreq%self.n_pitch_bins], bestFreq<self.n_pitch_bins)
        # breakpoint()
        # self.hmm_value, state = onlineViterbiState(self.hmm_value, np.log(observation_probs+self.tiny), self.log_trans[200,:401])
        if fast_matrix_mode:
            self.hmm_value, state = blockViterbiStateOpt(self.hmm_value, np.log(observation_probs+self.tiny), 
                self.log_trans_00, self.log_trans_01, self.log_trans_10, self.log_trans_11,
                self.n1, self.correspondence)
            # Find f0 corresponding to each decoded pitch bin.
            voiced_flag  = state < self.n1
            f0 = (self.freqs1[state] if voiced_flag else self.freqs2[state-self.n1])
        else:            
            self.hmm_value, state = onlineViterbiState(self.hmm_value, np.log(observation_probs+self.tiny), self.log_trans)
            # Find f0 corresponding to each decoded pitch bin.
            f0 = self.freqs[state % self.n_pitch_bins]
            voiced_flag = state < self.n_pitch_bins

        if not voiced_flag and self.fill_na is not None:
            f0 = self.fill_na

        return f0, voiced_flag, voiced_prob

def run_realtime_pyin_as_batch(
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
    lpyin = LivePyin(fmin, fmax, sr=22050, frame_length=2048,  
                 hop_length=None, n_thresholds=100, beta_parameters=(2, 18),
                 boltzmann_parameter=2, resolution=0.1, max_transition_rate=35.92,
                 switch_prob=0.01, no_trough_prob=0.01, fill_na=np.nan, 
                 dtype = y.dtype.type,
                 n_bins_per_semitone = 20, 
                 n_bins_per_unvoiced_semitone = 1, 
                 max_semitones_per_frame = 12,
                 transition_semitones_variance = None)
    if hop_length is None: hop_length = frame_length // 4 # Set the default hop if it is not already specified.
    y_frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length).T
    f0s, voiced_flags, voiced_probs = [],[],[]
    for yframe in y_frames:
        f0, voiced_flag, voiced_prob = lpyin.step(yframe)
        f0s.append(f0); voiced_flags.append(voiced_flag); voiced_probs.append(voiced_prob)
    return f0s, voiced_flags, voiced_probs

