"""Plots of the feature distributions of synthetic simulations against the biophysical simulation (Supplementary Fig. 6c)

Functions copied from notebooks/1_results/supp_fig6_CD_pros_of_added_details_yield_revised.ipynb.
"""

import numpy as np
from matplotlib import pyplot as plt


def plot_cell_to_site_distance(ax, distance, bins=20, title:str=None, ylim=(0,60), xlim=(0,25), edgecolor=(0.22, 0.5, 0.72), facecolor=(0.22, 0.5, 0.72), ylogscale=False, histtype='step', label=''):
    """
    Plots the firing rate distribution of units in a sorting extractor.

    Parameters:
    - firing rates: firing rates
    - bins: int, optional
        Number of bins for the histogram (default is 20).
    """
    ax.hist(distance, bins=bins, edgecolor=edgecolor, facecolor=facecolor, histtype=histtype, label=label)
    ax.set_xlabel("Cell-to-site distance (um)")
    ax.set_ylabel("Unit count", labelpad=1)
    if title:
        ax.set_title(title)
    if ylogscale:
        ax.set_yscale('log')
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)


def plot_firing_rate_distribution(ax, firing_rates, bins=20, title:str=None, ylim=(0,60), xlim=(0,25), edgecolor=(0.22, 0.5, 0.72), facecolor=(0.22, 0.5, 0.72), ylogscale=False, histtype='step'):
    """
    Plots the firing rate distribution of units in a sorting extractor.

    Parameters:
    - firing rates: firing rates
    - bins: int, optional
        Number of bins for the histogram (default is 20).
    """
    ax.hist(firing_rates, bins=bins, edgecolor=edgecolor, facecolor=facecolor, histtype=histtype)
    ax.set_xlabel("Firing rate (Hz)")
    ax.set_ylabel("Unit count")
    if title:
        ax.set_title(title)
    if ylogscale:
        ax.set_yscale('log')
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)


def plot_firing_ranges_distribution(ax, firing_ranges, bins=20, title:str='simulation_name', ylim=(0,50), xlim=(0,800), edgecolor=(0.22, 0.5, 0.72), facecolor=(0.22, 0.5, 0.72), ylogscale=False, histtype='step'):
    """
    Parameters:
    - amplitudes: waveform amplitudes
    - bins: int, optional
        Number of bins for the histogram (default is 20).
    """
    ax.hist(firing_ranges, bins=bins, edgecolor=edgecolor, facecolor=facecolor, histtype=histtype)
    ax.set_xlabel("Firing range (Hz)", labelpad=1)
    ax.set_ylabel("Unit count")
    if title:
        ax.set_title(title)
    if ylogscale:
        ax.set_yscale('log')
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)


def plot_waveform_amplitude_median_distribution(ax, amplitudes, bins=20, title:str='simulation_name', ylim=(0,50), xlim=(0,800), edgecolor=(0.22, 0.5, 0.72), facecolor=(0.22, 0.5, 0.72), ylogscale=False, histtype='step'):
    """
    Parameters:
    - amplitudes: waveform amplitudes
    - bins: int, optional
        Number of bins for the histogram (default is 20).
    """
    ax.hist(amplitudes, bins=bins, edgecolor=edgecolor, facecolor=facecolor, histtype=histtype)
    ax.set_xlabel("Amplitudes (uV)")
    ax.set_ylabel("Unit count")
    if title:
        ax.set_title(title)
    if ylogscale:
        ax.set_yscale('log')
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)


def plot_waveform_snr_distribution(ax, snrs, bins=20, title:str='simulation_name', ylim=(0,50), xlim=(0,800), edgecolor=(0.22, 0.5, 0.72), facecolor=(0.22, 0.5, 0.72), ylogscale=False, histtype='step'):
    """
    Parameters:
    - snrs: snrs
    - bins: int, optional
        Number of bins for the histogram (default is 20).
    """
    ax.hist(snrs, bins=bins, edgecolor=edgecolor, facecolor=facecolor, histtype=histtype)
    ax.set_xlabel("SNR")
    ax.set_ylabel("Unit count")
    if title:
        ax.set_title(title)
    if ylogscale:
        ax.set_yscale('log')
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)


def plot_spatial_spread_distribution(ax, mad_ratio, bins=20, title:str='simulation_name', ylim=(0,50), xlim=(0,800), edgecolor=(0.22, 0.5, 0.72), facecolor=(0.22, 0.5, 0.72), ylogscale=False, histtype='step'):
    """
    Parameters:
    - mad_ratio: mad_ratio
    - bins: int, optional
        Number of bins for the histogram (default is 20).
    """
    ax.hist(mad_ratio, bins=bins, edgecolor=edgecolor, facecolor=facecolor, histtype=histtype)
    ax.set_xlabel("Spatial spread (a.u)")
    ax.set_ylabel("Unit count")
    if title:
        ax.set_title(title)
    if ylogscale:
        ax.set_yscale('log')
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)   
    return ax


def plot_mad_ratio_distribution(ax, mad_ratios, bins=20, title:str='simulation_name', ylim=(0,50), xlim=(0,800), edgecolor=(0.22, 0.5, 0.72), facecolor=(0.22, 0.5, 0.72), ylogscale=False, histtype='step'):
    """
    Values close to 1 indicate a good isolation of the unit.
    i.e., the mean absolute deviation of the spike amplitude is similar to the mean absolute deviation 
    of the noise amplitude.
    
    Parameters:
    - mad_ratios: waveform mean-absolute deviation ratio between spike amplitude and background noise
    - bins: int, optional
        Number of bins for the histogram (default is 20).
    """
    ax.hist(mad_ratios, bins=bins, edgecolor=edgecolor, facecolor=facecolor, histtype=histtype)
    ax.set_xlabel("MAD ratios (a.u)")
    ax.set_ylabel("Unit count")
    if title:
        ax.set_title(title)
    if ylogscale:
        ax.set_yscale('log')
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)


def plot_spatial_metric(ax, metric, bins=20, title:str='simulation_name', ylim=(0,50), xlim=(0,800), edgecolor=(0.22, 0.5, 0.72), facecolor=(0.22, 0.5, 0.72), ylogscale=False, histtype='step', label=''):
    """
    Values close to 1 indicate a good isolation of the unit.
    i.e., the mean absolute deviation of the spike amplitude is similar to the mean absolute deviation 
    of the noise amplitude.
    
    Parameters:
    - mad_ratios: waveform mean-absolute deviation ratio between spike amplitude and background noise
    - bins: int, optional
        Number of bins for the histogram (default is 20).
    """
    ax.hist(metric, bins=bins, edgecolor=edgecolor, facecolor=facecolor, histtype=histtype, label=label)
    ax.set_ylabel("Unit count")
    if title:
        ax.set_title(title)
    if ylogscale:
        ax.set_yscale('log')
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)    
    return ax


def plot_cell_to_cell_distances(ax, cell_distance, bins=20, title:str='simulation_name', ylim=(0,50), xlim=(0,800), edgecolor=(0.22, 0.5, 0.72), facecolor=(0.22, 0.5, 0.72), ylogscale=False, histtype='step'):
    """
    Parameters:
    - cell_distance: all cell-to-cell euclidean distances
    - bins: int, optional
        Number of bins for the histogram (default is 20).
    """    
    ax.hist(cell_distance, bins=bins, edgecolor=edgecolor, facecolor=facecolor, histtype=histtype)
    ax.set_xlabel("Cell distances (um)")
    ax.set_ylabel("Unit count")
    if title:
        ax.set_title(title)
    if ylogscale:
        ax.set_yscale('log')
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)


def plot_features(axes, col_ix, features, features_ns, title: str, ylim=2000):

    # cell-to-electrode distance
    ax = axes[0,col_ix]
    plot_cell_to_site_distance(ax, features['cell_to_site_distance'], title=title, ylim=(0,210), xlim=(0,100), label='Synthetic model')
    plot_cell_to_site_distance(ax, features_ns['cell_to_site_distance_ns'], title='', ylim=(0,210), xlim=(0,100), edgecolor='r', facecolor='None', ylogscale=True, label='Biophysical model')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if col_ix!=0: ax.set_xlabel('')
    ax.set_title(title, pad=1.5)

    # cell-to-cell distance
    ax = axes[1,col_ix]
    plot_cell_to_cell_distances(ax, features['cell_distance'], title='', ylim=(0,210), xlim=(0,100))
    plot_cell_to_cell_distances(ax, features_ns['cell_distance_ns'], title='', ylim=(0,210), xlim=(0,100), edgecolor='r', facecolor='None', ylogscale=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if col_ix!=0: ax.set_xlabel('')

    # firing rate
    ax = axes[2,col_ix]
    plot_firing_rate_distribution(ax, features['firing_rates'], title='', ylim=(0,ylim), xlim=(0,25))
    plot_firing_rate_distribution(ax, features_ns['firing_rates_ns'], title='', ylim=(0,ylim), xlim=(0,25), edgecolor='r', facecolor='None', ylogscale=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if col_ix!=0: ax.set_xlabel('')

    # firing range
    ax = axes[3,col_ix]
    plot_firing_ranges_distribution(ax, features['firing_ranges'], title='', ylim=(0,ylim), xlim=(0,8))
    plot_firing_ranges_distribution(ax, features_ns['firing_ranges_ns'], title='', ylim=(0,ylim), xlim=(0,8), edgecolor='r', facecolor='None', ylogscale=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if col_ix!=0: ax.set_xlabel('')
    
    # cell waveform SNR
    ax = axes[4,col_ix]
    plot_waveform_snr_distribution(ax, features['snrs'], title='', ylim=(0,ylim), xlim=(0,80))
    plot_waveform_snr_distribution(ax, features_ns['snrs_ns'][np.isfinite(features_ns['snrs_ns'])], title='', ylim=(0,ylim), xlim=(0,80), edgecolor='r', facecolor='None', ylogscale=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if col_ix!=0: ax.set_xlabel('')

    # cell waveform amplitude
    ax = axes[5,col_ix]
    plot_waveform_amplitude_median_distribution(ax, features['amplitudes'], title='', ylim=(0,ylim), xlim=(-400, 40))
    plot_waveform_amplitude_median_distribution(ax, features_ns['amplitudes_ns'][np.isfinite(features_ns['amplitudes_ns'])], title='', ylim=(0,ylim), xlim=(-400,40), edgecolor='r', facecolor='None', ylogscale=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if col_ix!=0: ax.set_xlabel('')

    # cell waveform MAD ratio
    ax = axes[6,col_ix]
    plot_mad_ratio_distribution(ax, features['mad_ratios'], title='', ylim=(0,ylim), xlim=(0,15))
    plot_mad_ratio_distribution(ax, features_ns['mad_ratios_ns'][np.isfinite(features_ns['mad_ratios_ns'])], title='', ylim=(0,ylim), xlim=(0,15), edgecolor='r', facecolor='None', ylogscale=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if col_ix!=0: ax.set_xlabel('')

    # cell waveform peak_to_valley
    ax = axes[7,col_ix]
    ax = plot_spatial_metric(ax, np.array(features['peak_to_valley'])*1000, bins=20, title='', ylim=(0,ylim), xlim=(0,10))
    ax = plot_spatial_metric(ax, np.array(features_ns['peak_to_valley_ns'])*1000, bins=20, title='', ylim=(0,ylim), xlim=(0,10), edgecolor='r', facecolor='None', ylogscale=True, histtype='step')
    ax.set_xlabel('Peak-to-valley (ms)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)    
    if col_ix!=0: ax.set_xlabel('')

    # cell waveform repolarization_slope
    ax = axes[8, col_ix]
    ax = plot_spatial_metric(ax, np.array(features['repolarization_slope']), bins=20, title='', ylim=(0,3000), xlim=(0,4000000))
    ax = plot_spatial_metric(ax, np.array(features_ns['repolarization_slope_ns']), bins=20, title='', ylim=(0,3000), xlim=(0, 4000000), edgecolor='r', facecolor='None', ylogscale=True, histtype='step')
    ax.set_xlabel('Repolarization slope (uV/s)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if col_ix!=0: ax.set_xlabel('')

    # cell waveform repolarization_slope
    ax = axes[9, col_ix]
    ax = plot_spatial_metric(ax, np.array(features['exp_decay']), bins=20, title='', ylim=(0,ylim), xlim=(0,0.4))
    ax = plot_spatial_metric(ax, np.array(features_ns['exp_decay_ns']), bins=20, title='', ylim=(0,ylim), xlim=(0, 0.4), edgecolor='r', facecolor='None', ylogscale=True, histtype='step')
    ax.set_xlabel('Exponential decay (um)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)    
    if col_ix!=0: ax.set_xlabel('')

    # cell waveform spatial spread (custom)
    ax = axes[10,col_ix]
    plot_spatial_spread_distribution(ax, features['spreads_simi_weighted_distance'], title='', ylim=(0,ylim), xlim=(0,1200))
    plot_spatial_spread_distribution(ax, features_ns['spreads_simi_weighted_distance_ns'], title='', ylim=(0,ylim), xlim=(0,1200), edgecolor='r', facecolor='None', ylogscale=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)    
    ax.set_xlabel('Similarity-weighted spread (a.u)')
    if col_ix!=0: ax.set_xlabel('')

    # cell waveform spatial spread (spikeinterface)
    ax = axes[11,col_ix]
    ax = plot_spatial_metric(ax, np.array(features['spread']), bins=20, title='', ylim=(0,3000), xlim=(0,200))
    ax = plot_spatial_metric(ax, np.array(features_ns['si_spread_ns']), bins=20, title='', ylim=(0, 3000), xlim=(0, 200), edgecolor='r', facecolor='None', ylogscale=True, histtype='step')
    ax.set_xlabel('Spread (um)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)      
    if col_ix!=0: ax.set_xlabel('')

    # cell waveform velocity above
    ax = axes[12, col_ix]
    ax = plot_spatial_metric(ax, np.array(features['velocity_above']), bins=20, title='', ylim=(0,ylim), xlim=(-500,500))
    ax = plot_spatial_metric(ax, np.array(features_ns['velocity_above_ns']), bins=20, title='', ylim=(0,ylim), xlim=(-500,500), edgecolor='r', facecolor='None', ylogscale=True, histtype='step')
    ax.set_xlabel('Velocity above (um/s)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)    
    if col_ix!=0: ax.set_xlabel('')

    # cell waveform velocity below
    ax = axes[13, col_ix]
    ax = plot_spatial_metric(ax, np.array(features['velocity_below']), bins=20, title='', ylim=(0,ylim), xlim=(-500,500))
    ax = plot_spatial_metric(ax, np.array(features_ns['velocity_below_ns']), bins=20, title='', ylim=(0,ylim), xlim=(-500,500), edgecolor='r', facecolor='None', ylogscale=True, histtype='step')
    ax.set_xlabel('Velocity below (um/s)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)    
    if col_ix!=0: ax.set_xlabel('')    
    return axes

    
# Quality of unit isolation (single-unit yield)


# ROWS of Supplementary Fig. 6c: (feature, plot function, label, x-axis limits, y-axis limits (unit count), scale factor)
FEATURE_ROWS = [
    ("cell_to_site_distance", plot_cell_to_site_distance, "Cell-to-\nsite distance (um)", (0, 100), (0, 210), 1),
    ("cell_distance", plot_cell_to_cell_distances, "Cell distances (um)", (0, 100), (0, 210), 1),
    ("firing_rates", plot_firing_rate_distribution, "Firing rate (Hz)", (0, 25), (0, 2000), 1),
    ("firing_ranges", plot_firing_ranges_distribution, "Firing range (Hz)", (0, 8), (0, 2000), 1),
    ("snrs", plot_waveform_snr_distribution, "SNR", (0, 80), (0, 2000), 1),
    ("amplitudes", plot_waveform_amplitude_median_distribution, "Amplitudes (uV)", (-400, 40), (0, 2000), 1),
    ("mad_ratios", plot_mad_ratio_distribution, "MAD ratios (a.u)", (0, 15), (0, 2000), 1),
    ("peak_to_valley", plot_spatial_metric, "Peak-to-valley (ms)", (0, 10), (0, 2000), 1000),
    ("repolarization_slope", plot_spatial_metric, "Repolarization\nslope (uV/s)", (0, 4000000), (0, 3000), 1),
    ("exp_decay", plot_spatial_metric, "Exponential\ndecay (um)", (0, 0.4), (0, 2000), 1),
    ("spreads_simi_weighted_distance", plot_spatial_spread_distribution, "Spread (a.u)", (0, 1200), (0, 2000), 1),
    ("velocity_above", plot_spatial_metric, "Velocity above (um/s)", (-500, 500), (0, 2000), 1),
    ("velocity_below", plot_spatial_metric, "Velocity below (um/s)", (-500, 500), (0, 2000), 1),
]


def plot_feature_distributions(axes, col_ix, features, features_ns, title, n_matched=0):
    """Plot the distributions of the features of a synthetic simulation (blue) and of the biophysical simulation (red) in a column of axes

    Args:
        axes (np.array): axes (rows x columns)
        col_ix (int): column
        features (dict): features of the synthetic simulation (keys of FEATURE_ROWS)
        features_ns (dict): features of the biophysical simulation (same keys)
        title (str): title of the column
        n_matched (int): number of first rows (features) that the synthetic simulation matches (annotated 'Matched' by annotate_matched, after the layout)
    """
    for row, (key, plot_fn, _, xlim, ylim, scale) in enumerate(FEATURE_ROWS):
        ax = axes[row, col_ix]
        synthetic = np.asarray(features[key], dtype=float) * scale
        biophysical = np.asarray(features_ns[key], dtype=float) * scale
        plot_fn(ax, synthetic[np.isfinite(synthetic)], title="", ylim=ylim, xlim=xlim)
        plot_fn(ax, biophysical[np.isfinite(biophysical)], title="", ylim=ylim, xlim=xlim, edgecolor="r", facecolor="None", ylogscale=True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("")
        ax.set_ylabel("")
    axes[0, col_ix].set_title(title, fontsize=5.5, fontweight="bold", pad=11)
    return axes


def annotate_matched(ax, fontsize=6):
    """Write 'Matched' (red, same font size everywhere: 6 pt, within the 5-7 pt of the Nature guidelines) in the right of an axes
    where it does not overlap the histograms, above the axes if there is no room

    Call it after the layout of the figure is final.
    """
    renderer = ax.figure.canvas.get_renderer()
    paths = [p.get_path().transformed(p.get_transform()) for p in ax.patches] + [l.get_path().transformed(l.get_transform()) for l in ax.lines]
    for y in (0.98, 0.84, 0.70, 0.56, 0.42, 0.28):
        text = ax.text(0.995, y, "Matched", transform=ax.transAxes, ha="right", va="top", fontsize=fontsize, color="red")
        box = text.get_window_extent(renderer)
        if not any(path.intersects_bbox(box, filled=False) for path in paths):
            return text
        text.remove()
    return ax.text(0.995, 1.04, "Matched", transform=ax.transAxes, ha="right", va="bottom", fontsize=fontsize, color="red")


MODEL_TITLES = {
    "buccino": "Buccino model (2020)",
    "s2": "Mearec model 2 (S2)",
    "s3": "Mearec model 3 (S3)",
    "s4": "Mearec model 4 (S4)",
    "s5": "Mearec model 5 (S5)",
}
N_MATCHED = {"buccino": 0, "s2": 2, "s3": 3, "s4": 4, "s5": 5}  # number of first features that the model is constrained to match


def plot_features_figure(features: dict, features_ns: dict, fig_size=(6.6, 5.6)):
    """Supplementary Fig. 6c: distributions of the features of the synthetic simulations (blue) and of our biophysical simulation (red)

    Args:
        features (dict): {model: {feature: array}} for the models of MODEL_TITLES
        features_ns (dict): {feature: array} of the biophysical simulation
        fig_size (tuple): figure size in inches

    Returns:
        fig, axes
    """
    models = list(MODEL_TITLES)
    fig, axes = plt.subplots(len(FEATURE_ROWS), len(models), figsize=fig_size)
    for col, model in enumerate(models):
        plot_feature_distributions(axes, col, features[model], features_ns, MODEL_TITLES[model], n_matched=N_MATCHED[model])
    for ax in axes.ravel():
        ax.tick_params(labelsize=5.5, length=2.5, pad=1)
    plt.subplots_adjust(left=0.19, right=0.99, top=0.925, bottom=0.035, wspace=0.3, hspace=1.1)
    for col, model in enumerate(models):
        for row in range(N_MATCHED[model]):
            annotate_matched(axes[row, col])

    # row labels and common labels
    for row, (_, _, label, _, _, _) in enumerate(FEATURE_ROWS):
        pos = axes[row, 0].get_position()
        fig.text(0.145, (pos.y0 + pos.y1) / 2, label, ha="right", va="center", fontsize=5.5, fontweight="bold", family="Arial")
    fig.text(0.02, 0.5, "Unit count", rotation=90, ha="center", va="center", fontsize=6, family="Arial")
    fig.text(0.6, 0.985, "Increasingly detailed MEAREC models vs our model", ha="center", va="top", fontsize=7, fontweight="bold", family="Arial")
    return fig, axes
