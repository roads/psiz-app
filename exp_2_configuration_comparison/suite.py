"""Experiment 2: Trial configuration comparison.

Comparing different trial configurations using simulations.

An initial embedding is inferred using real observations data. This
embedding is then treated as ground truth for the purpose of simulating
human behavior for two different display configurations.

Notes:
    For simiplicity, the dimensionality is not inferred at each step.
        Instead, the correct dimensionality is provided.
    Perform multiple runs.

"""

import copy
import itertools

import numpy as np
from scipy.stats import sem
import matplotlib
import matplotlib.pyplot as plt
import pickle
from psiz.models import Exponential, load_embedding
from psiz.dimensionality import suggest_dimensionality
from psiz.simulate import Agent
from psiz.generator import RandomGenerator
from psiz import datasets
from psiz.utils import similarity_matrix, matrix_correlation


def main():
    """Run experiment 2."""
    # Settings.
    n_run = 3
    emb_filepath = '/Users/bdroads/Projects/psiz-app/exp_2_configuration_comparison/emb_true.hdf5'
    data_2c1_rand_filepath = '/Users/bdroads/Projects/psiz-app/exp_2_configuration_comparison/data_2c1_rand.p'
    data_8c2_rand_filepath = '/Users/bdroads/Projects/psiz-app/exp_2_configuration_comparison/data_8c2_rand.p'
    figure_path = '/Users/bdroads/Projects/psiz-app/exp_2_configuration_comparison/results.pdf'

    dataset_name = 'birds-16'
    (obs, catalog) = datasets.load_dataset(dataset_name)
    time_s_2c1 = 3.06  # TODO compute from obs
    time_s_8c2 = 8.98  # TODO compute from obs

    cond_info_2c1 = {
        'name': 'Random 2-choose-1',
        'n_reference': 2,
        'n_select': 1,
        'n_trial_initial': 500,
        'n_trial_total': 80500,  # 50500
        'n_trial_per_round': 80000,  # 10000,
        'time_s_per_trial': time_s_2c1
    }
    cond_info_8c2 = {
        'name': 'Random 8-choose-2',
        'n_reference': 8,
        'n_select': 2,
        'n_trial_initial': 250,
        'n_trial_total': 25250,
        'n_trial_per_round': 5000,
        'time_s_per_trial': time_s_8c2
    }
    freeze_options = {'theta': {'rho': 2}}

    # Infer a ground truth embedding from real observations.
    # emb_true = real_embedding(obs, catalog, freeze_options)
    # emb_true.save(emb_filepath)

    emb_true = load_embedding(emb_filepath)
    data_2c1_rand = multiple_runs_random(
        n_run, emb_true, cond_info_2c1, freeze_options)
    pickle.dump(data_2c1_rand, open(data_2c1_rand_filepath, 'wb'))

    # data_8c2_rand = multiple_runs_random(
    #     n_run, emb_true, cond_info_8c2, freeze_options)
    # pickle.dump(data_8c2_rand, open(data_8c2_rand_filepath, 'wb'))

    # data_2c1_rand = pickle.load(open(data_2c1_rand_filepath, 'rb'))
    # data_8c2_rand = pickle.load(open(data_8c2_rand_filepath, 'rb'))
    # plot_results((data_2c1_rand, data_8c2_rand), figure_path)


def real_embedding(obs, catalog, freeze_options):
    """Fit embedding model to real observations."""
    # Determine dimensionality.
    n_dim = suggest_dimensionality(
        obs, Exponential, catalog.n_stimuli, freeze_options=freeze_options,
        verbose=1)
    # Determine embedding using all available observation data.
    emb_true = Exponential(catalog.n_stimuli, n_dim)
    emb_true.freeze(freeze_options)
    emb_true.fit(obs, n_restart=40, verbose=2)
    return emb_true


def multiple_runs_random(n_run, emb_true, cond_info, freeze_options):
    """Perform multiple runs of simulation."""
    results = None
    for _ in range(n_run):
        results = concat_runs(
            results,
            simulate_random_condition(emb_true, cond_info, freeze_options)
        )
    data_rand = {'info': cond_info, 'results': results}
    return data_rand


def simulate_random_condition(emb_true, cond_info, freeze_options):
    """Simulate progress for a particular trial configuration.

    Record:
        n_trial, loss, R^2
    """
    # Similarity matrix associated with true embedding.
    simmat_true = similarity_matrix(
        emb_true.similarity, emb_true.z['value'])

    # Generate a random docket of trials.
    generator = RandomGenerator(emb_true.n_stimuli)
    docket = generator.generate(
        cond_info['n_trial_total'],
        cond_info['n_reference'],
        cond_info['n_select']
    )

    # Simulate similarity judgments.
    agent = Agent(emb_true)
    obs = agent.simulate(docket)

    # Infer independent models with increasing amounts of data.
    n_trial = np.arange(
        cond_info['n_trial_initial'],
        cond_info['n_trial_total'] + 1,
        cond_info['n_trial_per_round']
    )
    n_round = len(n_trial)
    r_squared = np.empty((n_round))
    loss = np.empty((n_round))
    for i_round in range(n_round):
        # Infer embedding.
        emb_inferred = Exponential(emb_true.n_stimuli, emb_true.n_dim)
        emb_inferred.freeze(freeze_options)
        include_idx = np.arange(0, n_trial[i_round])
        loss[i_round] = emb_inferred.fit(
            obs.subset(include_idx), n_restart=20)
        # Compare the inferred model with ground truth by comparing the
        # similarity matrices implied by each model.
        simmat_infer = similarity_matrix(
            emb_inferred.similarity, emb_inferred.z['value'])
        r_squared[i_round] = matrix_correlation(simmat_infer, simmat_true)
        print('Round {0} | R^2 {1:.2f}'.format(i_round, r_squared[i_round]))

    results = {
        'n_trial': np.expand_dims(n_trial, axis=1),
        'loss': np.expand_dims(loss, axis=1),
        'r_squared': np.expand_dims(r_squared, axis=1)
    }
    return results


def concat_runs(results, results_new):
    """Concatenate information from different runs."""
    if results is None:
        results = results_new
    else:
        for key in results:
            results[key] = np.concatenate(
                (results[key], results_new[key]), axis=1
            )
    return results


def plot_results(results, figure_path):
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
        name = results[condition]['info']['name']
        time_cost_hr = (
            results[condition]['info']['time_s_per_trial'] *
            np.mean(results[condition]['results']['n_trial'], axis=1) /
            3600
        )
        # loss = np.mean(results[condition]['loss'], axis=1)
        r_squared = results[condition]['results']['r_squared']
        r_squared_mean = np.mean(r_squared, axis=1)
        r_squared_sem = sem(r_squared, axis=1)

        ax.plot(
            time_cost_hr, r_squared_mean, '-', color=c_line[i_cond],
            label=name)
        ax.fill_between(
            time_cost_hr,
            r_squared_mean - r_squared_sem,
            r_squared_mean + r_squared_sem,
            color=c_env[i_cond]
        )
        # Add text at .9 R^2 breakpoint.
        locs = np.greater_equal(r_squared_mean, .9)
        if np.sum(locs) > 0:
            time_thresh = time_cost_hr[locs]
            time_thresh = time_thresh[0]
            r2_thresh = r_squared_mean[locs]
            r2_thresh = r2_thresh[0]
            ax.scatter(
                time_thresh, r2_thresh, marker='d', color=c_scatter[i_cond],
                edgecolors='k')
            ax.text(
                time_thresh, r2_thresh, "{0:.1f} hr".format(time_thresh),
                fontdict=fontdict)

    ax.set_ylim(bottom=0., top=1.)
    ax.set_xlabel('Total Worker Hours')
    ax.set_ylabel(r'$R^2$ Similarity')
    ax.legend()
    plt.tight_layout()

    if figure_path is None:
        plt.show()
    else:
        plt.savefig(
            figure_path, format='pdf', bbox_inches="tight", dpi=100)


if __name__ == "__main__":
    main()
