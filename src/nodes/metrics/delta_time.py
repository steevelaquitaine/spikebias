"""Sorting accuracy as a function of the delta time of the comparison between sorted and ground truth spikes"""

import numpy as np
import spikeinterface as si
from spikeinterface import comparison
from joblib import Parallel, delayed


def get_unit_accuracies(gt_path: str, sorted_path: str, delta_time: float, duration_sec: float = 600):
    """Accuracy of each ground truth unit (best agreement score with a sorted unit)

    Args:
        gt_path (str): path of the ground truth SortingExtractor
        sorted_path (str): path of the sorted SortingExtractor
        delta_time (float): time window (ms) around each ground truth spike within which sorted and ground truth spikes coincide
        duration_sec (float): duration of the ground truth analysed, from the start of the recording

    Returns:
        np.array: accuracy of each ground truth unit, in descending order
    """
    SortingTrue = si.load_extractor(gt_path)
    SortingTrue = SortingTrue.frame_slice(start_frame=0, end_frame=int(duration_sec * SortingTrue.get_sampling_frequency()))
    Sorting = si.load_extractor(sorted_path)
    comp = comparison.compare_sorter_to_ground_truth(SortingTrue, Sorting, exhaustive_gt=True, delta_time=delta_time)
    return comp.agreement_scores.max(axis=1).sort_values(ascending=False).values


def get_accuracy_vs_delta_time(gt_path: str, sorted_path: str, delta_times, duration_sec: float = 600, threshold: float = 0.8, n_jobs: int = 16):
    """Sorting accuracy (proportion of ground truth units with an accuracy of at least `threshold`) for each delta time

    Returns:
        np.array: sorting accuracy for each delta time
    """
    def one(dt):
        acc = get_unit_accuracies(gt_path, sorted_path, dt, duration_sec)
        return np.sum(acc >= threshold) / len(acc)

    return np.array(Parallel(n_jobs=n_jobs)(delayed(one)(dt) for dt in delta_times))


def get_accuracy_vs_shift(gt_path: str, sorted_path: str, unit_id, delta_time: float, shifts_us, duration_sec: float = 600):
    """Accuracy of a ground truth unit after constant shifts of its spike timestamps

    The accuracy is the best agreement score between the shifted ground truth unit and the sorted units.
    A positive shift subtracts the shift from the timestamps, as spikeinterface's align_sorting does.

    Args:
        gt_path (str): path of the ground truth SortingExtractor
        sorted_path (str): path of the sorted SortingExtractor
        unit_id: id of the ground truth unit
        delta_time (float): delta time (ms) of the comparison
        shifts_us (array): shifts (microseconds)
        duration_sec (float): duration analysed, from the start of the recording

    Returns:
        np.array: accuracy for each shift
    """
    from spikeinterface.core import NumpySorting

    Gt = si.load_extractor(gt_path)
    sfreq = Gt.get_sampling_frequency()
    Gt = Gt.frame_slice(start_frame=0, end_frame=int(duration_sec * sfreq))
    Sorting = si.load_extractor(sorted_path)
    spikes = Gt.get_unit_spike_train(unit_id)
    accs = []
    for shift_us in shifts_us:
        shift_smp = 0 if np.abs(shift_us) < 1e6 / sfreq else int(shift_us * sfreq / 1e6)       # no shift below the sampling period
        shifted = NumpySorting.from_unit_dict([{unit_id: spikes - shift_smp}], sfreq)
        comp = comparison.compare_sorter_to_ground_truth(shifted, Sorting, exhaustive_gt=True, delta_time=delta_time)
        accs.append(comp.agreement_scores.values.max())
    return np.array(accs)


def get_accuracy_vs_shift_all(gt_path: str, sorted_path: str, unit_ids, delta_times, shifts_us, duration_sec: float = 600, n_jobs: int = 16):
    """Accuracy of each ground truth unit after constant shifts of its spike timestamps, for each delta time

    Returns:
        dict: {delta time: {unit id: np.array of accuracies, one for each shift}}
    """
    pairs = [(dt, u) for dt in delta_times for u in unit_ids]
    out = Parallel(n_jobs=n_jobs)(delayed(get_accuracy_vs_shift)(gt_path, sorted_path, u, dt, shifts_us, duration_sec) for dt, u in pairs)
    result = {dt: {} for dt in delta_times}
    for (dt, u), accs in zip(pairs, out):
        result[dt][u] = accs
    return result
