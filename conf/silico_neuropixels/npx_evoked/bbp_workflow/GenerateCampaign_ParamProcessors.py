# Description:   BBP-WORKFLOW parameter processor functions used to generate SSCx simulation campaigns
# Author:        C. Pokorny
# Date:          17/02/2021
# Last modified: 28/09/2021

import os
import shutil
import numpy as np
import pandas as pd
from bluepy import Circuit
import lookup_projection_locations as projloc
import stimulus_generation as stgen
import hashlib
import json
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
import math
from scipy.stats import linregress


""" Generates user target file, combining targets from custom list and projection paths """
def generate_user_target(*, circuit_config, path, custom_user_targets=[], **kwargs):
    
    circuit = Circuit(circuit_config)
    proj_paths = list(circuit.config['projections'].values())
    proj_targets = [os.path.join(os.path.split(p)[0], 'user.target') for p in proj_paths]
    proj_targets = list(filter(os.path.exists, proj_targets))
    
    target_paths = custom_user_targets + proj_targets
    
    user_target_name = 'user.target'
    user_target_file = os.path.join(path, user_target_name)
    with open(user_target_file, 'w') as f_tgt:
        for p in target_paths:
            assert os.path.exists(p), f'ERROR: Target "{p}" not found!'
            with open(p, 'r') as f_src:
                f_tgt.write(f_src.read())
                f_tgt.write('\n\n')
            # print(f'INFO: Adding target "{p}" to "{os.path.join(os.path.split(path)[-1], user_target_name)}"')
    
    # Set group membership to same as <path> (should be 10067/"bbp")
    # os.chown(user_target_file, uid=-1, gid=os.stat(path).st_gid)
    
    # print(f'INFO: Generated user target "{os.path.join(os.path.split(path)[-1], user_target_name)}"')
    
    return {'user_target_name': user_target_name}


""" Places stimulation file from template into simulation folders """
def stim_file_from_template(*, path, stim_file_template, **kwargs):
    
    stim_filename = 'input.dat'
    stim_file = os.path.join(path, stim_filename)
    shutil.copyfile(stim_file_template, stim_file)
    
    # print(f'INFO: Added stim file from template to {stim_file}')
    
    return {'stim_file': stim_filename}


""" Sets user-defined (e.g., layer-specific) shot noise levels based on scaling factors and method """
def apply_shot_noise_scaling(*, shotn_scale_method, shotn_scale_factors, **kwargs):
    
    assert np.all([tgt in kwargs for tgt in shotn_scale_method.keys()]), 'ERROR: Scale target error!'
    
    # Scaling function, depending on method
    # [Comparison of methods: git@bbpgitlab.epfl.ch/conn/structural/dendritic_synapse_density/MissingSynapses.ipynb]
    def scale_fct(value, scale, method):
        if method == 'none': # No scaling
            scaled_value = value
        elif method == 'linear': # Linear scaling
            scaled_value = value * scale
        elif method == 'linear_bounded': # Linear scaling, bounded to max=100.0
            max_value = 100.0
            scaled_value = np.minimum(value * scale, max_value)
        elif method == 'exponential': # Exponential scaling, converging to max=100.0
            max_value = 100.0
            tau = -1 / np.log(1 - value / max_value)
            scaled_value = max_value * (1 - np.exp(-scale / tau))
        else:
            assert False, 'ERROR: Scale method unknown!'
        # return np.round(scaled_value).astype(int)
        return scaled_value
    
    # Apply scaling
    shotn_scale_dict = {}
    for spec, scale in shotn_scale_factors.items(): # Specifier and scaling factor, e.g. "L1I": 0.6
        for tgt in shotn_scale_method.keys(): # Scaling target, e.g. "shotn_mean_pct"
            shotn_scale_dict.update({f'{tgt}_{spec}': scale_fct(kwargs[tgt], scale, shotn_scale_method[tgt])})    
    
    return shotn_scale_dict


def calculate_suggested_unconnected_firing_rate(target_connected_fr, a, b, c):

    y = target_connected_fr
    log_domain = max((y - c) / a, 1.0)
    # print(y, c, a, log_domain)
    suggested_unconnected_fr = math.log(log_domain) / b

    return suggested_unconnected_fr



def fit_exponential(ca_stat1, ca_stat2):

    popt, pcov = curve_fit(
        lambda t, a, b, c: a * np.exp(b * t) + c,
        ca_stat1, ca_stat2, p0=(1.0, 0.5, ca_stat2.min() - 1), 
        maxfev=20000
    )

    return popt



def create_delete_flag(path):

    delete_flag_file = os.path.join(path, 'DELETE_FLAG.FLAG')
    f = open(delete_flag_file, "x")
    f.close()



def set_conductance_scalings_for_desired_frs(*, path, ca, depol_stdev_mean_ratio, desired_connected_proportion_of_invivo_frs, in_vivo_reference_frs, data_for_unconnected_fit_name, data_for_connected_adjustment_fit_name, unconnected_connected_fr_adjustment_fit_method, **kwargs):

    scaling_and_data_neuron_class_keys = {
    
    "L1I":"L1_INH",
    "L23E":"L23_EXC",
    "L23I":"L23_INH",
    "L4E":"L4_EXC",
    "L4I":"L4_INH",
    "L5E":"L5_EXC",
    "L5I":"L5_INH",
    "L6E":"L6_EXC",
    "L6I":"L6_INH"
    
    }

    should_create_delete_flag = False
    
    unconn_df = pd.read_parquet(path=data_for_unconnected_fit_name)

    if (data_for_connected_adjustment_fit_name != ''):
        unconn_conn_df = pd.read_parquet(path=data_for_connected_adjustment_fit_name)
        # ca_unconn_conn_df = unconn_conn_df.etl.q(ca=ca, window='conn_spont')
        ca_unconn_conn_df = unconn_conn_df[(unconn_conn_df['ca']==ca) & (unconn_conn_df['depol_stdev_mean_ratio']==depol_stdev_mean_ratio) & (unconn_conn_df["window"] == 'conn_spont') & (unconn_conn_df['bursting'] == False)]

        ca_unconn_conn_df["invivo_fr"] = ca_unconn_conn_df['desired_connected_fr'] / ca_unconn_conn_df['desired_connected_proportion_of_invivo_frs']
        ca_unconn_conn_df = ca_unconn_conn_df[ca_unconn_conn_df['mean_of_mean_firing_rates_per_second'] < ca_unconn_conn_df['invivo_fr'] * 1.05]


    scale_dict = {}
    for scaling_neuron_class_key, in_vivo_fr in in_vivo_reference_frs.items():

        neuron_class = scaling_and_data_neuron_class_keys[scaling_neuron_class_key]

        in_vivo_fr = in_vivo_reference_frs[scaling_neuron_class_key]
        desired_connected_fr = desired_connected_proportion_of_invivo_frs*in_vivo_fr

        if (data_for_connected_adjustment_fit_name != ''):

            nc_ca_unconn_conn_df = ca_unconn_conn_df[ca_unconn_conn_df['neuron_class']==neuron_class]

            if (nc_ca_unconn_conn_df[nc_ca_unconn_conn_df["depol_stdev_mean_ratio"] == depol_stdev_mean_ratio]['mean_of_mean_firing_rates_per_second'].max() < desired_connected_fr):
                should_create_delete_flag = True

            unconnected_connected_fr_adjustment_fit_method = 'exponential'

            if (unconnected_connected_fr_adjustment_fit_method == 'exponential'):            

                popt = fit_exponential(nc_ca_unconn_conn_df['desired_unconnected_fr'], nc_ca_unconn_conn_df['mean_of_mean_firing_rates_per_second'])
                desired_unconnected_fr = calculate_suggested_unconnected_firing_rate(desired_connected_fr, popt[0], popt[1], popt[2])

            elif (unconnected_connected_fr_adjustment_fit_method == 'linear'):

                ca_lr = linregress(ca_stat1, ca_stat2)
                ca_lr_slope = np.around(ca_lr.slope, 3)

                desired_unconnected_fr = nc_ca_unconn_conn_df['desired_unconnected_fr'] / ca_lr_slope

            # print(ca, neuron_class, desired_connected_fr, desired_unconnected_fr)
        
        else: 
            desired_unconnected_fr = desired_connected_fr



        nc_unconn_df = unconn_df[(unconn_df["ca"] == 1.15) & (unconn_df["neuron_class"] == scaling_and_data_neuron_class_keys[scaling_neuron_class_key])]
        
        # print(ng_data_for_plot)
        gradient_line_x = np.linspace(0,100,1000)
        gradient_line_y = gradient_line_x * depol_stdev_mean_ratio

        predicted_frs_for_line = griddata(nc_unconn_df[['mean', 'stdev']].to_numpy(), nc_unconn_df["data"], (gradient_line_x, gradient_line_y), method='cubic')
        
        # print(scaling_neuron_class_key, predicted_frs_for_line, desired_unconnected_fr)
        index_of_closest_point = np.nanargmin(abs(predicted_frs_for_line - desired_unconnected_fr))

        closest_x = round(gradient_line_x[index_of_closest_point], 3)
        closest_y = round(gradient_line_y[index_of_closest_point], 3)

        scale_dict.update({f'desired_connected_fr_{scaling_neuron_class_key}': desired_connected_fr})
        scale_dict.update({f'desired_unconnected_fr_{scaling_neuron_class_key}': desired_unconnected_fr})
        scale_dict.update({f'ornstein_uhlenbeck_mean_pct_{scaling_neuron_class_key}': closest_x})
        scale_dict.update({f'ornstein_uhlenbeck_sd_pct_{scaling_neuron_class_key}': closest_y})
        # scale_dict.update({f'shotn_mean_pct_{scaling_neuron_class_key}': closest_x})
        # scale_dict.update({f'shotn_sd_pct_{scaling_neuron_class_key}': closest_y})

        # print(scale_dict)

    if (should_create_delete_flag):
        create_delete_flag(path)

    return scale_dict




def get_cfg_hash(cfg):
    """
    Generates MD5 hash code for given config dict
    """
    hash_obj = hashlib.md5()

    def sort_dict(data):
        """
        Sort dict entries recursively, so that the hash function
        does not depend on the order of elements
        """
        if isinstance(data, dict):
            return {key: sort_dict(data[key]) for key in sorted(data.keys())}
        else:
            return data

    # Generate hash code from sequence of cfg params (keys & values)
    for k, v in sort_dict(cfg).items():
        hash_obj.update(str(k).encode('UTF-8'))
        hash_obj.update(str(v).encode('UTF-8'))
    
    return hash_obj.hexdigest()





import spikewriter

# Originally created by Andras Ecker
def _load_circuit_targets(circuit_path, user_target_path):
    """Loads circuit and adds targets from user defined user.target"""
    c = Circuit(circuit_path)
    if os.path.exists(user_target_path):
        c_cfg = c.config.copy()
        c_cfg["targets"].append(user_target_path)
        c = Circuit(c_cfg)  # reload circuit with extra targets from user.target
    return c

# Adapted by James Isbister
def _load_multiple_circuit_targets(circuit_path, user_target_paths):
    """Loads circuit and adds targets from user defined user.target"""
    c = Circuit(circuit_path)
    c_cfg = c.config.copy()
    for user_target_path in user_target_paths:
        if os.path.exists(user_target_path):
            c_cfg["targets"].append(user_target_path)
    c = Circuit(c_cfg)  # reload circuit with extra targets from user.target
    return c

# Originally created by Andras Ecker
def _get_projection_locations(path, c, proj_name, mask, supersample):
    """Local helper to avoid looping `projloc.get_projection_locations()` by saving the results
    and next time loading the saved results instead of recalculating the whole thing again"""
    supersample_str = "__supersample" if supersample else ""
    save_name = os.path.join(os.path.split(path)[0], "projections", "%s__%s%s.txt" % (proj_name, mask, supersample_str))
    if not os.path.isfile(save_name):
        gids, pos2d, pos3d, _ = projloc.get_projection_locations(c, proj_name, mask=mask,
                                                                 mask_type="dist", supersample=supersample)
        pos = pos2d if pos2d is not None else pos3d
        if not os.path.isdir(os.path.dirname(save_name)):
            os.mkdir(os.path.dirname(save_name))
        np.savetxt(save_name, np.concatenate((gids.reshape(-1, 1), pos), axis=1))
    else:
        tmp = np.loadtxt(save_name)
        # print(save_name, tmp)
        gids, pos = tmp[:, 0].astype(int), tmp[:, 1:]
    return gids, pos

# Adapted from code created by Andras Ecker
def gen_whisker_flick_stim_and_find_fibers_all(*, path, **kwargs):

    param_list = ["circuit_config", "circuit_target", "user_target_path",  # base circuit
                  "stim_seed", "vpm_pct", "pom_pct", # structural
                  "stim_delay", "num_stims", "inter_stimulus_interval",  # stim. series
                  # "vpm_mu", "pom_mu", "vpm_sigma", "pom_sigma", "vpm_spike_rate", "pom_spike_rate",
                  "vpm_proj_name", "pom_proj_name", "supersample", "data_for_vpm_input"
                  ]  # spikes

    cfg = {p: kwargs.get(p) for p in param_list}
    # Load circuit and fix targets
    c = _load_circuit_targets(cfg["circuit_config"], cfg["user_target_path"])

    all_spike_times = []; all_spiking_gids = []
    fib_grps = ['vpm']
    for fib_grp_i, fib_grp in enumerate(fib_grps):

        gids, pos = _get_projection_locations(path, c, cfg[fib_grp + "_proj_name"], "hex0", cfg["supersample"])
        centre_point = np.mean(pos, axis=0)

        new_gids = gids
        new_pos = pos            

        np.random.seed(cfg["stim_seed"] + fib_grp_i)

        pct_key = fib_grp + "_pct"

        if (cfg[pct_key] > 0.0):
            selected_gid_indices = np.sort(np.random.choice(range(len(new_gids)), size=int(len(new_gids) * cfg[pct_key]/100.), replace=False))
            selected_gids = new_gids[selected_gid_indices]
            selected_gid_poss = new_pos[selected_gid_indices]

            plt.figure()
            plt.scatter(pos[:, 0], pos[:, 1])
            plt.scatter(np.asarray(new_pos)[:, 0], np.asarray(new_pos)[:, 1])
            plt.scatter(np.asarray(selected_gid_poss)[:, 0], np.asarray(selected_gid_poss)[:, 1])
            plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig(os.path.join(path, fib_grp + '_selected_stim_fibers.png'))
            plt.close()

            stim_times = spikewriter.generate_stim_series_num_stims(cfg["stim_delay"], cfg["num_stims"], cfg["inter_stimulus_interval"])

            # spike_times, spiking_gid_indices = spikewriter.generate_lognormal_spike_train(stim_times, selected_gid_indices,
            #                                         cfg[fib_grp + "_mu"], cfg[fib_grp + "_sigma"], cfg[fib_grp + "_spike_rate"], cfg["stim_seed"] + fib_grp_i)

            # spike_times, spiking_gid_indices = spikewriter.generate_yu_svoboda_spike_trains(stim_times, selected_gid_indices, cfg["data_for_vpm_input"], cfg["stim_seed"] + fib_grp_i)
            spike_times, spiking_gid_indices = spikewriter.generate_ji_diamond_estimate_scaled_spike_trains(stim_times, selected_gid_indices, cfg["stim_seed"] + fib_grp_i)

            spiking_gids = gids[spiking_gid_indices]

            plt.figure()
            plt.scatter(spike_times, spiking_gid_indices)
            plt.gca().set_xlim([1500, 1550])
            plt.savefig(os.path.join(path, fib_grp + 'stim_spikes.png'))
            plt.close()

            all_spike_times += spike_times.tolist()
            all_spiking_gids += spiking_gids.tolist()

        # Write to file and return file name used in the template
    stim_file = "input.dat"
    spikewriter.write_spikes(all_spike_times, all_spiking_gids, os.path.join(path, stim_file))

    return {"stim_file": stim_file}


def line_gradient_and_intersect_for_angle(angle_degrees, centre_point):



    if (angle_degrees > 0.0) & (angle_degrees < 90.0):
        gradient = 1.0 / math.tan(math.radians(angle_degrees))

    elif (angle_degrees > 90.0) & (angle_degrees < 180.0):
        gradient = -math.tan(math.radians(angle_degrees - 90.0))

    elif (angle_degrees > 180.0) & (angle_degrees < 270.0):
        gradient = 1.0 / math.tan(math.radians(angle_degrees - 180.0))

    elif (angle_degrees > 270.0) & (angle_degrees < 360.0):
        gradient = -math.tan(math.radians(angle_degrees - 270.0))
        

    elif angle_degrees == 0.0:
        gradient = np.nan

    elif angle_degrees == 90.0:
        gradient = 0.0

    elif angle_degrees == 180.0:
        gradient = np.nan

    elif angle_degrees == 270.0:
        gradient = 0.0
        
    intersection = centre_point[1] - (gradient * centre_point[0])


    return gradient, intersection

def within_trailing_edge(p, angle, gradient, intersection, centre_point):

    if (angle != 0.0) and (angle != 180):

        diff_to_line = gradient * p[0] + intersection - p[1]

        if angle < 180.0:
            if diff_to_line > 0.0:
                return True

        if angle > 180.0:
            if diff_to_line < 0.0:
                return True

    else:
        if angle == 0.0:
            if p[0] > centre_point[0]:
                return True

        elif angle == 180.0:
            if p[0] < centre_point[0]:
                return True

    return False

def within_leading_edge(p, angle, gradient, intersection, centre_point):

    if (angle != 0.0) and (angle != 180):

        diff_to_line = gradient * p[0] + intersection - p[1]

        if angle < 180.0:
            if diff_to_line < 0.0:
                return True

        if angle > 180.0:
            if diff_to_line > 0.0:
                return True
    else:
        if angle == 0.0:
            if p[0] < centre_point[0]:
                return True

        elif angle == 180.0:
            if p[0] > centre_point[0]:
                return True




    return False





# Adapted from code created by Andras Ecker
def gen_whisker_flick_stim_and_find_fibers_pizza_slice(*, path, **kwargs):
    """
    Generates whisker step (longer then flick) like VPM and optionally POm (on top) spike trains
    VPM fibers are clustered together (to e.g. scan toposample stim. params.) and as clustering happens on the fly,
    it's a more general version of `gen_whisker_flick_stim()` that can be applied to any version and region of the SSCx
    """

    param_list = ["circuit_config", "circuit_target", "user_target_path",  # base circuit
                  "stim_seed", "vpm_num_fibres", "pom_pct", # structural
                  "stim_delay", "num_trials_per_stimulus", "inter_stimulus_interval",  # stim. series
                  "vpm_mu", "pom_mu", "vpm_sigma", "pom_sigma", "vpm_spike_rate", "pom_spike_rate",
                  "vpm_proj_name", "pom_proj_name", "supersample",  # spikes
                  "number_of_pizza_slices", "rotations"]

    cfg = {p: kwargs.get(p) for p in param_list}
    # Load circuit and fix targets
    # c = _load_circuit_targets(cfg["circuit_config"], )
    c = _load_multiple_circuit_targets(cfg["circuit_config"], [cfg["user_target_path"], "/gpfs/bbp.cscs.ch/project/proj83/home/bolanos/Bernstein2022/network/concentric_targets/cyls_hex0.target"])

    all_spike_times = []; all_spiking_gids = []
    fib_grps = ['vpm']
    all_rotations = []; all_presentation_times = []
    for fib_grp_i, fib_grp in enumerate(fib_grps):

        # gids, pos = _get_projection_locations(path, c, cfg[fib_grp + "_proj_name"], cfg["circuit_target"], cfg["supersample"])
        gids, pos = _get_projection_locations(path, c, cfg[fib_grp + "_proj_name"], "cyl_hex0_0.035", cfg["supersample"])

        if (fib_grp == "vpm"):

            centre_point = np.mean(pos, axis=0)
            pizza_slice_degree = 360.0 / float(cfg["number_of_pizza_slices"])
            np.random.seed(cfg["stim_seed"] + fib_grp_i)

            spike_times, spiking_gid_indices = spikewriter.generate_ji_diamond_estimate_scaled_spike_trains([0], gids, cfg["stim_seed"] + fib_grp_i)

            # print(spike_times)
            # print(spiking_gid_indices)


            for rotation_index, rotation in enumerate(np.fromstring(cfg["rotations"], dtype=float, sep=',')):

                pizza_slice_leading_edge_degree = (pizza_slice_degree / 2.0 + rotation) % 360.0
                pizza_slice_trailing_edge_degree = ((pizza_slice_degree / 2.0) - pizza_slice_degree + rotation) % 360.0

                gradient_leading_edge, intersection_leading_edge = line_gradient_and_intersect_for_angle(pizza_slice_leading_edge_degree, centre_point)
                gradient_trailing_edge, intersection_trailing_edge = line_gradient_and_intersect_for_angle(pizza_slice_trailing_edge_degree, centre_point)

                new_pos = []; new_gids = [];
                for p, gid in zip(pos, gids):
                    if within_trailing_edge(p, pizza_slice_trailing_edge_degree, gradient_trailing_edge, intersection_trailing_edge, centre_point) and within_leading_edge(p, pizza_slice_leading_edge_degree, gradient_leading_edge, intersection_leading_edge, centre_point):
                        new_pos.append(p)
                        new_gids.append(gid)
                
                new_gids = np.asarray(new_gids)
                new_pos = np.asarray(new_pos)

                FULL_TARGET_gids, FULL_TARGET_pos = _get_projection_locations(path, c, cfg[fib_grp + "_proj_name"], cfg["circuit_target"], cfg["supersample"])

                plt.figure()
                plt.scatter(FULL_TARGET_pos[:, 0], FULL_TARGET_pos[:, 1], c='r')
                plt.scatter(pos[:, 0], pos[:, 1])
                plt.scatter(new_pos[:, 0], new_pos[:, 1])
                # plt.plot(FULL_TARGET_pos[:, 0], FULL_TARGET_pos[:, 0] * gradient_leading_edge + intersection_leading_edge, c='cyan')
                # plt.plot(FULL_TARGET_pos[:, 0], FULL_TARGET_pos[:, 0] * gradient_trailing_edge + intersection_trailing_edge, c='green')
                plt.gca().set_aspect('equal', adjustable='box')
                plt.savefig(os.path.join(path, fib_grp + '_PIZZA_stim_fibers_' + str(rotation) + '.png'))
                plt.close()


        
                num_fibres_key = fib_grp + "_num_fibres"

                if (cfg[num_fibres_key] > 0.0):

                    num_new_gids = len(new_gids)
                    num_new_gids_to_use = int(np.min([num_new_gids, cfg[num_fibres_key]]))

                    # sample_type = "RANDOM"
                    sample_type = "SMALLEST GIDS"

                    if sample_type == "RANDOM":
                        selected_gid_indices = np.sort(np.random.choice(range(num_new_gids), size=num_new_gids_to_use, replace=False))
                        selected_gids = new_gids[selected_gid_indices]
                        selected_gid_poss = new_pos[selected_gid_indices]

                    elif sample_type == "SMALLEST GIDS":

                        args_sorted_new_gids = np.argsort(new_gids)
                        selected_gids = new_gids[args_sorted_new_gids[:num_new_gids_to_use]]
                        selected_gid_poss = new_pos[args_sorted_new_gids[:num_new_gids_to_use]]



                    plt.figure()
                    plt.scatter(np.asarray(new_pos)[:, 0], np.asarray(new_pos)[:, 1])
                    plt.scatter(np.asarray(selected_gid_poss)[:, 0], np.asarray(selected_gid_poss)[:, 1])
                    plt.gca().set_aspect('equal', adjustable='box')
                    plt.savefig(os.path.join(path, fib_grp + '_selected_stim_fibers' + str(rotation) + '.png'))
                    plt.close()

                    stimulus_first_trial_time = cfg["stim_delay"] + rotation_index * cfg["num_trials_per_stimulus"] * cfg["inter_stimulus_interval"]

                    stim_times = spikewriter.generate_stim_series_num_stims(stimulus_first_trial_time, cfg["num_trials_per_stimulus"], cfg["inter_stimulus_interval"])

                    all_rotations.extend([rotation for t in range(cfg["num_trials_per_stimulus"])])
                    all_presentation_times.extend(stim_times)

                    for gid in selected_gids:
                        gid_spike_time = spike_times[np.argwhere(spiking_gid_indices == gid)[0]]

                        for stim_time in stim_times:

                            all_spike_times += [gid_spike_time + stim_time]
                            all_spiking_gids += [gid]

    #                 # spike_times, spiking_gid_indices = spikewriter.generate_lognormal_spike_train(stim_times, selected_gid_indices,
    #                 #                                         cfg[fib_grp + "_mu"], cfg[fib_grp + "_sigma"], cfg[fib_grp + "_spike_rate"], cfg["stim_seed"] + fib_grp_i)

    #                 spike_times, spiking_gid_indices = spikewriter.generate_ji_diamond_estimate_scaled_spike_trains(stim_times, selected_gid_indices, cfg["stim_seed"] + fib_grp_i)
    #                 spiking_gids = gids[spiking_gid_indices]


                    # plt.figure()
                    # plt.scatter(spike_times, spiking_gid_indices)
                    # plt.gca().set_xlim([stimulus_first_trial_time, stimulus_first_trial_time + 100])
                    # plt.savefig(os.path.join(path, fib_grp + 'stim_spikes' + str(rotation) + '.png'))
                    # plt.close()

                    # all_spike_times += spike_times.tolist()
                    # all_spiking_gids += spiking_gids.tolist()

        # Write to file and return file name used in the template
    stim_file = "input.dat"
    spikewriter.write_spikes(all_spike_times, all_spiking_gids, os.path.join(path, stim_file))

    rotations_by_presentation_time_file = "rotations_by_presentation_time.dat"
    spikewriter.write_spikes(all_rotations, all_presentation_times, os.path.join(path, rotations_by_presentation_time_file))


    blueetl_windows_str = f""
    for rotation_index, rotation in enumerate(np.fromstring(cfg["rotations"], dtype=float, sep=',')):

        initial_offset = cfg["stim_delay"] + rotation_index * cfg["num_trials_per_stimulus"] * cfg["inter_stimulus_interval"]
        n_trials = cfg["num_trials_per_stimulus"]
        trial_steps_value = cfg["inter_stimulus_interval"]
        window_type = "evoked_stimulus_onset_zeroed"

        windows = ["evoked_SOZ_25ms"]
        boundss = [[0, 25]]
        
        for window, bounds in zip(windows, boundss):

            window_name = window + "_" + str(rotation)
            blueetl_windows_str += f"        {window_name}:\n          bounds: {bounds}\n          initial_offset: {initial_offset}\n          n_trials: {n_trials}\n          trial_steps_value: {trial_steps_value}\n          window_type: {window_type}\n"

    text_file = open(os.path.join(path, "blueetl_windows.txt"), "w")
    n = text_file.write(blueetl_windows_str)
    text_file.close()


    return {"stim_file": stim_file}

