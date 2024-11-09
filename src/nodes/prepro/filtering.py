import numpy as np 
import logging
import logging.config
import yaml
from scipy.signal import detrend
from sklearn.preprocessing import MinMaxScaler
import pywt
from concurrent.futures import ProcessPoolExecutor
import spikeinterface.extractors as se 
import spikeinterface.preprocessing as spre

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


def std(trace, nlevel=6):
    """Estimates the standard deviation of the input trace for rescaling
    the Wavelet's coefficients.

    Returns
        Standard deviation of the input trace as (1D ndarray)
    """
    sigma = np.array([1.4825 * np.median(np.abs(trace[i])) for i in range(nlevel)])
    return sigma


def get_han_threshold(trace: np.array, sigma: np.array, coeffs: np.array, nlevels: int):

    # count samples
    num_samples = len(trace)

    # han et al threshold
    details_threshs = np.array([np.nan] * len(coeffs[1:]))

    # threshold for first detail coeff d_i=0
    details_threshs[0] = sigma[1] * np.sqrt(2 * np.log(num_samples))

    # threshold from 1 < d_i < NLEVELS
    for d_i in range(1, nlevels - 1):
        details_threshs[d_i] = (sigma[d_i] * np.sqrt(2 * np.log(num_samples))) / np.log(
            d_i + 1
        )
    # threhsold for d_i = nlevels
    details_threshs[nlevels - 1] = (
        sigma[nlevels - 1] * np.sqrt(2 * np.log(num_samples))
    ) / np.sqrt(nlevels - 1)
    return details_threshs


def wavelet_filter_single_trace(trace, num_samples:int, sfreq:int, wavelet:str='haar', method:str='hard', nlevel:int=6):
    """Denoise and high-pass filter a single trace with wavelet filtering
    with the 'han' threshold definition

    Args:
        trace (np.array): 1d voltage trace array 
        num_samples (int): trace's number of samples
        wavelet (str, optional): _description_. Defaults to 'haar'.
        method (str, optional): _description_. Defaults to 'hard'.
        nlevel (int, optional): _description_. Defaults to 6.

    Returns:
        np.array: denoised trace
    """

    # # signal acquisition parameters
    nyquist = sfreq/2
    num_samples = int(num_samples)
    
    # print cutoff frequency
    # a decomposition level of 6 produces an frequency cutoff of Fc = 234.375 Hz.
    freq_cutoff = nyquist / 2**nlevel  # cutoff frequency (the max of lowest freq. band)
    logger.info(f"high-pass filt. cutoff frequency: {freq_cutoff} Hz")
    
    # detrend and normalize
    detrended = detrend(trace)
    scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
    detrended = scaler.fit_transform(detrended.reshape(-1, 1))[:, 0]
    normalized = detrended.copy()


    # ********* Wavelet filtering *******************

    # make sure number of samples is a multiple of 2**NLEVEL
    size = (normalized.shape[0] // 2**nlevel) * (2**nlevel)
    logger.info(f"number of samples used for decomposition: {size}")

    # initialize filter
    wavelet = pywt.Wavelet(wavelet)

    # translation-invariance modification of the Discrete Wavelet Transform
    # that does not decimate coefficients at every transformation level.
    coeffs = pywt.swt(
        normalized[:size], wavelet, level=nlevel, start_level=0, trim_approx=True
    )

    # ********** Denoising ************************

    # estimate the wavelet coefficients std
    sigma = std(coeffs[1:], nlevel=nlevel)

    # determine the thresholds of the coefficients per level ('han')
    threshs = get_han_threshold(
        trace=trace,
        sigma=sigma,
        coeffs=coeffs,
        nlevels=nlevel,
    )    

    # apply the thresholds to the detail coeffs
    coeffs[1:] = [
        pywt.threshold(coeff_i, value=threshs[i], mode=method)
        for i, coeff_i in enumerate(coeffs[1:])
    ]
    logger.info("denoising done.")

    # ********** High-pass filtering ************************

    # clear approximation coefficients (set to 0)
    coeffs[0] = np.zeros(len(coeffs[0]))

    # sanity check
    assert sum(coeffs[0]) == 0, "approximation coeffs not cleared"
    logger.info("high-pass filtering done.")

    # ********* Reconstruct and reverse normalize

    # reconstruct, reverse normalize and remove DC component
    denoised_trace = pywt.iswt(coeffs, wavelet)
    denoised_trace = scaler.inverse_transform(denoised_trace.reshape(-1, 1))[:, 0]
    denoised_trace -= np.mean(denoised_trace)
    logger.info("reconstruction done.")
    return denoised_trace


def wavelet_filter_wrapper(trace, site, args):
    """parallelizes wavelet filtering on the computer cores by 
    wavelet_filter_parallelized()

    Args:
        trace (np.array): 1d trace array
        site (int): silent variable
        args (dict): filtering parameters

    Returns:
        _type_: _description_
    """
    return wavelet_filter_single_trace(trace,
                          num_samples=args["num_samples"],
                          sfreq=args["sfreq"],
                          wavelet=args["wavelet"],
                          method=args["method"],
                          nlevel=args["nlevel"]
                          )


def wavelet_filter(RecordingExtractor, 
                   duration_s:int, 
                   wavelet:str, 
                   method:str, 
                   nlevel:int):
    """_summary_

    Args:
        RecordingExtractor (_type_): 
        num_samples (int): number of trace samples
        sfreq (int): _description_
        wavelet (str): _description_
        method (str): _description_
        nlevel (int): _description_

    Returns:
        RecordingExtractor: int16 traces
    """
    # convert to int16 
    # Kilosort sorters assume int16 traces
    # It enables 8X faster sorting than float64.
    RecordingExtractor = spre.astype(RecordingExtractor, "int16")

    # encapsulate the filtering parameters
    args = dict()
    sfreq = RecordingExtractor.get_sampling_frequency()
    args["num_samples"] = int(sfreq * duration_s)
    args["sfreq"] = sfreq
    args["wavelet"] = wavelet
    args["method"] = method
    args["nlevel"] = nlevel

    # traces
    traces = RecordingExtractor.get_traces()
    nsites = traces.shape[1]
    
    # parallelize filtering over sites
    with ProcessPoolExecutor() as executor:
        denoised_trace = executor.map(
            wavelet_filter_wrapper,
            traces.T,
            np.arange(0, nsites, 1),
            [args]*nsites, # repeat parameter set for each core
        )
    denoised_traces = list(denoised_trace)
    denoised_traces = np.array(denoised_traces).T

    # build into a RecordingExtractor
    # and copy the initial extractor's 
    # attributes
    Prepro = se.NumpyRecording(
        traces_list=[denoised_traces],
        sampling_frequency=sfreq,
    )
    RecordingExtractor.copy_metadata(Prepro)

    # enforce int16
    RecordingExtractor = spre.astype(RecordingExtractor, "int16")
    return Prepro