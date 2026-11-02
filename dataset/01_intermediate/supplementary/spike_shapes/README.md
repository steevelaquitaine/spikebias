# Spike shapes above and below the somata (Supplementary Fig. 3a and 3b)

`spike_shapes_above_below_somata.csv`: spike-triggered extracellular voltage of 16 excitatory (panel a) and 20 inhibitory (panel b) model neurons located within 50 um of the Neuropixels probe, recorded by an electrode 20 um above and an electrode 20 um below the soma, from 1 ms before to 3 ms after the spike peak (40 kHz, 160 samples). For each morphological type (mtype) the neuron with the highest firing rate is shown.

Columns: panel (a excitatory, b inhibitory), synapse_class, column and row (position in the figure), mtype, electrode (above, below), time_ms, voltage_mV.

## Provenance (read this)

The original values come from the point-source LFP report of the circuit simulation (`neuropixels_lfp_10m_384ch_hex0_40Khz_2023_10_18/selectedNeurons`, BlueConfig, bluepy; notebook `notebooks/1_results/supp_fig3_AB_plotSpikes.ipynb`), which is not part of this repository. These values were reconstructed from the vector figure that this notebook produced (`figures/6_supp/fig3/fig3A_spikes_ex_mtypes.svg`, `fig3B_spikes_inh_mtypes.svg`): vertices of each curve converted back to mV and ms with the axes' tick labels, and linearly interpolated on the 0.025 ms sampling grid. Matplotlib had simplified the drawn curves, so values are exact to about 0.1 pixel of the original figure, not to the simulation's floating-point precision. To obtain the exact values, re-run the notebook with the simulation output and export `lfpDict` for the 36 selected cells.

Code that generated the CSV: `src/pipes/validation/spike_shapes/extract_spike_shapes_from_svg.py` (reads the two svg files of git commit 546e1cb; if they are deleted, check them out of that commit first).
