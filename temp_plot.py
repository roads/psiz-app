import os
import copy
import itertools
from pathlib import Path

import numpy as np
from numpy import ma
from scipy.stats import sem
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pickle

def main(fp_results):
    fp_data_r2c1 = fp_results / Path('exp_2/birds/r2c1/r2c1_data.p')
    fp_data_r8c2 = fp_results / Path('exp_2/birds/r8c2/r8c2_data.p')
    fp_data_a8c2 = fp_results / Path('exp_2/birds/a8c2/a8c2_785_data.p')
    fp_figure_exp2b = fp_results / Path('exp_2/birds/exp2_temp.pdf')
    
    data_r2c1 = pickle.load(open(fp_data_r2c1, 'rb'))
    data_r8c2 = pickle.load(open(fp_data_r8c2, 'rb'))
    data_a8c2 = pickle.load(open(fp_data_a8c2, 'rb'))

    # results = data_a8c2['results']
    # results_r1 = data_a8c2_r1['results']
    # results = add_temp_results(results, results_r1)
    # fp_data_a8c2 = fp_results / Path('exp_2/birds/a8c2/a8c2_data_new.p')
    # pickle.dump(data_a8c2, open(fp_data_a8c2, 'wb'))

    # Standard.
    rgb1 = np.array((0.0, 0.0, 0.5312, 1.0))
    rgb2 = np.array((1.0, 0.8125, 0.0, 1.0))
    rgb3 = np.array((0.5, 0.0, 0.0, 1.0))
    # Transparent version.
    alpha_val = .2
    rgb1_trans = np.array((0.0, 0.0, 0.5312, alpha_val))
    rgb2_trans = np.array((1.0, 0.8125, 0.0, alpha_val))
    rgb3_trans = np.array((0.5, 0.0, 0.0, alpha_val))
    # Lighter version.
    color_scale = .4  # Lower scale yeilds lighter colors.
    rgb1_light = 1 - (color_scale * (1 - rgb1))
    rgb2_light = 1 - (color_scale * (1 - rgb2))
    rgb3_light = 1 - (color_scale * (1 - rgb3))

    c_line = [tuple(rgb1), tuple(rgb3), tuple(rgb2)]
    # c_env = [tuple(rgb1_light), tuple(rgb3_light)]
    c_env = [tuple(rgb1_trans), tuple(rgb3_trans), tuple(rgb2_trans)]
    c_scatter = [
        np.expand_dims(rgb1, axis=0),
        np.expand_dims(rgb3, axis=0),
        np.expand_dims(rgb2, axis=0)
    ]

    fontdict = {
        'fontsize': 10,
        'verticalalignment': 'top',
        'horizontalalignment': 'left'
    }

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    i_cond = 2
    plot_condition(ax, data_r2c1, c_line[i_cond], c_env[i_cond], c_scatter[i_cond], fontdict)

    i_cond = 0
    plot_condition(ax, data_r8c2, c_line[i_cond], c_env[i_cond], c_scatter[i_cond], fontdict)

    i_cond = 1
    plot_condition(ax, data_a8c2, c_line[i_cond], c_env[i_cond], c_scatter[i_cond], fontdict)
    
    # f = data_a8c2['info']['time_s_per_trial'] / (60 * 60)
    # time_cost_hr = data_a8c2['results']['n_trial'][:, 0] * f
    # r_squared_mean = data_a8c2['results']['r_squared'][:, 0]
    # ax.plot(
    #     time_cost_hr, r_squared_mean, '-', color=c_line[i_cond],
    #     label='Active 1')
    
    # f = data_a8c2_r5['info']['time_s_per_trial'] / (60 * 60)
    # time_cost_hr = data_a8c2_r5['results']['n_trial'][:, 0] * f
    # r_squared_mean = data_a8c2_r5['results']['r_squared'][:, 0]
    # time_cost_hr = time_cost_hr[0:-108]
    # r_squared_mean = r_squared_mean[0:-108]
    # ax.plot(
    #     time_cost_hr, r_squared_mean, '-', color=c_line[i_cond],
    #     label='Active 5')

    ax.set_ylim(bottom=0., top=1.)
    ax.set_xlabel('Total Worker Hours')
    ax.set_ylabel(r'Pearson $\rho$')
    ax.legend()
    plt.tight_layout()

    plt.savefig(
        fp_figure_exp2b.absolute().as_posix(), format='pdf',
        bbox_inches="tight", dpi=100)


def add_temp_results(results, results_new):
    for key in results:
        n_entry = results[key].shape[0]
        n_run_old = results[key].shape[1]
        n_entry_new = results_new[key].shape[0]
        if n_entry > n_entry_new:
            filler = np.zeros([n_entry - n_entry_new, 1], dtype=results_new[key].dtype)
            results_new[key] = np.concatenate((results_new[key], filler), axis=0)
        elif n_entry_new > n_entry:
            filler = np.zeros([n_entry_new - n_entry, n_run_old], dtype=results_new[key].dtype)
            results[key] = np.concatenate((results[key], filler), axis=0)
        results[key] = np.concatenate(
            (results_new[key], results[key]), axis=1
        )
    return results


def plot_condition(ax, data, c_line, c_env, c_scatter, fontdict, rsquared_crit=.95):
    """Plot condition."""
    legend_name = data['info']['name']
    results = data['results']

    time_factor = data['info']['time_s_per_trial'] / (60 * 60)
    n_run = results['n_trial'].shape[1]
    
    # TODO
    if 'is_valid' in results:
        is_valid = results['is_valid']
    else:
        is_valid = np.ones(results['r_squared'].shape, dtype=bool)

    n_trial_mask = ma.array(results['n_trial'], mask=np.logical_not(is_valid))
    n_trial_avg = np.mean(n_trial_mask, axis=1)
    
    r_squared_mask = ma.array(results['r_squared'], mask=np.logical_not(is_valid)) ** (.5)
    
    r_squared_mean_avg = np.mean(r_squared_mask, axis=1)
    r_squared_mean_min = np.min(r_squared_mask, axis=1)
    r_squared_mean_max = np.max(r_squared_mask, axis=1)

    ax.semilogx(
        time_factor * n_trial_avg, r_squared_mean_avg, '-', color=c_line,
        label='{0:s} ({1:d})'.format(legend_name, n_run))
    ax.fill_between(
        time_factor * n_trial_avg, r_squared_mean_min, r_squared_mean_max,
        facecolor=c_env, edgecolor='none'
    )

    # Plot Criterion
    dmy_idx = np.arange(len(r_squared_mean_avg))
    locs = np.greater_equal(r_squared_mean_avg, rsquared_crit)
    if np.sum(locs) > 0:
        after_idx = dmy_idx[locs]
        after_idx = after_idx[0]
        before_idx = after_idx - 1
        segment_rise = r_squared_mean_avg[after_idx] - r_squared_mean_avg[before_idx]
        segment_run = n_trial_avg[after_idx] - n_trial_avg[before_idx]
        segment_slope = segment_rise / segment_run
        xg = np.arange(segment_run, dtype=np.int)
        yg = segment_slope * xg + r_squared_mean_avg[before_idx]

        locs2 = np.greater_equal(yg, rsquared_crit)
        trial_thresh = xg[locs2]
        trial_thresh = trial_thresh[0]
        r2_thresh = yg[trial_thresh]
        trial_thresh = trial_thresh + n_trial_avg[before_idx]

        # time_thresh = n_trial_avg[locs]
        # time_thresh = time_thresh[0]
        # r2_thresh = r_squared_mean_avg[locs]
        # r2_thresh = r2_thresh[0]
        ax.scatter(
            time_factor * trial_thresh, r2_thresh, marker='d', color=c_scatter,
            edgecolors='k')
        ax.text(
            time_factor * trial_thresh, r2_thresh + .06, "{0:.1f}".format(time_factor * trial_thresh),
            fontdict=fontdict)


if __name__ == "__main__":
    # fp_results = Path('/Users/bdroads/Projects/psiz-app/results')
    # fp_results = Path('/home/brett/packages/psiz-app/results')
    fp_results = Path('/home/brett/Projects/psiz-app.git/results')
    main(fp_results)