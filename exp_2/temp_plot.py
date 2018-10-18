import os
import copy
import itertools
from pathlib import Path

import numpy as np
from scipy.stats import sem
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pickle

def main(results_path):
    fp_data_r2c1 = results_path / Path('exp_2/data_r2c1.p')
    fp_data_r8c2 = results_path / Path('exp_2/data_r8c2_2d.p')
    fp_data_a8c2 = results_path / Path('exp_2/data_a8c2.p_temp')
    fp_figure_exp2b = results_path / Path('exp_2/exp2b.pdf')

    data_r8c2 = pickle.load(open(fp_data_r8c2, 'rb'))
    data_a8c2 = pickle.load(open(fp_data_a8c2, 'rb'))

    time_s_8c2 = 8.98

    rgb1 = np.array((0.0, 0.0, 0.5312, 1.0))
    rgb3 = np.array((0.5, 0.0, 0.0, 1.0))
    # Lighter version.
    color_scale = .4  # Lower scale yeilds lighter colors.
    rgb1_light = 1 - (color_scale * (1 - rgb1))
    rgb3_light = 1 - (color_scale * (1 - rgb3))

    c_line = [tuple(rgb1), tuple(rgb3)]
    c_env = [tuple(rgb1_light), tuple(rgb3_light)]

    # plot_exp2((data_r8c2, data_a8c2), fp_figure_exp2b)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    f = 1  # time_s_8c2 / (60 * 60)
    # time_cost_hr_avg = np.mean(data_r8c2['results']['n_trial'] * f, axis=1)
    # r_squared_mean_avg = np.mean(data_r8c2['results']['r_squared'], axis=1)
    # r_squared_mean_min = np.min(data_r8c2['results']['r_squared'], axis=1)
    # r_squared_mean_max = np.max(data_r8c2['results']['r_squared'], axis=1)
    # ax.plot(
    #     time_cost_hr_avg, r_squared_mean_avg, '-', color=c_line[0],
    #     label='Random')
    # ax.fill_between(
    #     time_cost_hr_avg,
    #     r_squared_mean_min,
    #     r_squared_mean_max,
    #     color=c_env[0]
    # )

    time_cost_hr = data_r8c2['results']['n_trial'][:, 0] * f
    r_squared_mean = data_r8c2['results']['r_squared'][:, 0]
    ax.plot(
        time_cost_hr, r_squared_mean, '-', color=c_line[0],
        label='Random 1')

    # time_cost_hr = data_r8c2['results']['n_trial'][:, 1] * f
    # r_squared_mean = data_r8c2['results']['r_squared'][:, 1]
    # ax.plot(
    #     time_cost_hr, r_squared_mean, '-', color='r',
    #     label='Random 2')

    # time_cost_hr = data_r8c2['results']['n_trial'][:, 2] * f
    # r_squared_mean = data_r8c2['results']['r_squared'][:, 2]
    # ax.plot(
    #     time_cost_hr, r_squared_mean, '-', color='r',
    #     label='Random 3')

    # time_cost_hr = data_r8c2['results']['n_trial'][:, 3] * f
    # r_squared_mean = data_r8c2['results']['r_squared'][:, 3]
    # ax.plot(
    #     time_cost_hr, r_squared_mean, '-', color='r',
    #     label='Random 4')

    # time_cost_hr = data_r8c2['results']['n_trial'][:, 4] * f
    # r_squared_mean = data_r8c2['results']['r_squared'][:, 4]
    # ax.plot(
    #     time_cost_hr, r_squared_mean, '-', color='r',
    #     label='Random 5')

    n_round = len(data_a8c2['results']['n_trial'])
    # max_round_idx = n_round - 1
    # plot_idx = np.arange(0, n_round, 10)
    # plot_idx = np.hstack((plot_idx, np.array([max_round_idx])))
    # plot_idx = np.unique(plot_idx)
    plot_idx = np.arange(0, n_round)
    time_cost_hr = data_a8c2['results']['n_trial'][plot_idx] * f
    r_squared_mean = data_a8c2['results']['r_squared'][plot_idx]
    ax.plot(
        time_cost_hr, r_squared_mean, '-', color=c_line[1],
        label='Active 1')

    ax.set_ylim(bottom=0., top=1.)
    ax.set_xlabel('Trials')
    ax.set_ylabel(r'$R^2$ Similarity')
    ax.legend()
    plt.tight_layout()

    plt.savefig(
        fp_figure_exp2b.absolute().as_posix(), format='pdf',
        bbox_inches="tight", dpi=100)


def plot_exp2(results, fp_figure):
    """Visualize results of experiment."""
    fontdict = {
        'fontsize': 10,
        'verticalalignment': 'top',
        'horizontalalignment': 'left'
    }

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    rgb1 = np.array((0.0, 0.0, 0.5312, 1.0))
    # rgb2 = np.array((1.0, 0.8125, 0.0, 1.0))
    rgb3 = np.array((0.5, 0.0, 0.0, 1.0))
    # Lighter version.
    color_scale = .4  # Lower scale yeilds lighter colors.
    rgb1_light = 1 - (color_scale * (1 - rgb1))
    # rgb2_light = 1 - (color_scale * (1 - rgb2))
    rgb3_light = 1 - (color_scale * (1 - rgb3))

    c_line = [tuple(rgb1), tuple(rgb3)]
    c_env = [tuple(rgb1_light), tuple(rgb3_light)]
    c_scatter = [
        np.expand_dims(rgb1, axis=0),
        np.expand_dims(rgb3, axis=0)
    ]

    # Compute statistics across runs for each condition (on the fly).
    for i_cond, condition in enumerate(results):
        name = condition['info']['name']
        time_cost_hr = (
            condition['info']['time_s_per_trial'] *
            np.mean(condition['results']['n_trial'], axis=1) /
            3600
        )
        # loss = np.mean(condition['loss'], axis=1)
        r_squared = condition['results']['r_squared']
        r_squared_mean = np.mean(r_squared, axis=1)
        r_squared_sem = sem(r_squared, axis=1)

        ax.plot(
            time_cost_hr, r_squared_mean, '-', color=c_line[i_cond],
            label=name)
        # ax.fill_between(
        #     time_cost_hr,
        #     r_squared_mean - r_squared_sem,
        #     r_squared_mean + r_squared_sem,
        #     color=c_env[i_cond]
        # )
        # Add text at .9 R^2 breakpoint.
        # locs = np.greater_equal(r_squared_mean, .9)
        # if np.sum(locs) > 0:
        #     time_thresh = time_cost_hr[locs]
        #     time_thresh = time_thresh[0]
        #     r2_thresh = r_squared_mean[locs]
        #     r2_thresh = r2_thresh[0]
        #     ax.scatter(
        #         time_thresh, r2_thresh, marker='d', color=c_scatter[i_cond],
        #         edgecolors='k')
        #     ax.text(
        #         time_thresh, r2_thresh + .06, "{0:.1f} hr".format(time_thresh),
        #         fontdict=fontdict)

    ax.set_ylim(bottom=0., top=1.)
    ax.set_xlabel('Total Worker Hours')
    ax.set_ylabel(r'$R^2$ Similarity')
    ax.legend()
    plt.tight_layout()

    if fp_figure is None:
        plt.show()
    else:
        plt.savefig(
            fp_figure.absolute().as_posix(), format='pdf',
            bbox_inches="tight", dpi=100)

if __name__ == "__main__":
    # results_path = Path('/Users/bdroads/Projects/psiz-app/results')
    # results_path = Path('/home/brett/packages/psiz-app/results')
    results_path = Path('/home/brett/Projects/psiz-app.git/results')
    main(results_path)