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

import os
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
from psiz import visualize
from psiz.utils import similarity_matrix, matrix_correlation


def experiment_2(results_path):
    """Run experiment 2."""
    # Settings.
    dataset_name = 'birds-16'
    # seed_list = [913, 192, 785, 891, 841]
    seed_list = [913]

    emb_filepath = os.path.join(results_path, 'emb_true_3d.hdf5')
    data_2c1_rand_filepath = os.path.join(results_path, 'data_2c1_rand.p')
    data_8c2_rand_filepath = os.path.join(results_path, 'data_8c2_rand.p')
    figure_path = os.path.join(results_path, 'exp2.pdf')

    (obs, catalog) = datasets.load_dataset(dataset_name)
    time_s_2c1 = 3.06  # TODO compute from obs
    time_s_8c2 = 8.98  # TODO compute from obs

    cond_info_2c1 = {
        'name': 'Random 2-choose-1',
        'n_reference': 2,
        'n_select': 1,
        'n_trial_initial': 500,
        'n_trial_total': 150500,
        'n_trial_per_round': 5000,
        'time_s_per_trial': time_s_2c1
    }
    cond_info_8c2 = {
        'name': 'Random 8-choose-2',
        'n_reference': 8,
        'n_select': 2,
        'n_trial_initial':  250,
        'n_trial_total': 15250,
        'n_trial_per_round': 500,
        'time_s_per_trial': time_s_8c2
    }
    freeze_options = {'theta': {'rho': 2, 'beta': 10}}

    # Infer a ground truth embedding from real observations.
    # np.random.seed(123)
    # emb_true = real_embedding(obs, catalog, freeze_options)
    # emb_true.save(emb_filepath)

    emb_true = load_embedding(emb_filepath)
    # visualize.visualize_embedding_static(
    #     emb_true.z['value'], class_vec=catalog.stimuli.class_id.values,
    #     classes=catalog.class_label, filename=figure_path)  # TODO

    # data_2c1_rand = None
    # data_2c1_rand = pickle.load(open(data_2c1_rand_filepath, 'rb'))
    # data_2c1_rand = multiple_runs_random(
    #     seed_list, emb_true, cond_info_2c1, freeze_options, data_2c1_rand)
    # pickle.dump(data_2c1_rand, open(data_2c1_rand_filepath, 'wb'))

    data_8c2_rand = None
    # data_8c2_rand = pickle.load(open(data_8c2_rand_filepath, 'rb'))
    data_8c2_rand = multiple_runs_random(
        seed_list, emb_true, cond_info_8c2, freeze_options, data_8c2_rand)
    pickle.dump(data_8c2_rand, open(data_8c2_rand_filepath, 'wb'))

    data_2c1_rand = pickle.load(open(data_2c1_rand_filepath, 'rb'))
    data_8c2_rand = pickle.load(open(data_8c2_rand_filepath, 'rb'))
    plot_exp2((data_2c1_rand, data_8c2_rand), figure_path)


def real_embedding(obs, catalog, freeze_options):
    """Fit embedding model to real observations."""
    # Determine dimensionality.
    n_dim = suggest_dimensionality(
        obs, Exponential, catalog.n_stimuli, freeze_options=freeze_options,
        verbose=1)
    # Determine embedding using all available observation data.
    emb_true = Exponential(catalog.n_stimuli, n_dim)
    emb_true.freeze(freeze_options)
    emb_true.fit(obs, n_restart=40, verbose=3)
    return emb_true


def multiple_runs_random(
        seed_list, emb_true, cond_info, freeze_options, data_rand=None):
    """Perform multiple runs of simulation.

    Note: Random number generator is seeded before the beginning of
        each run.
    """
    if data_rand is not None:
        results = data_rand['results']
    else:
        results = None

    n_run = len(seed_list)
    for i_seed in range(n_run):
        np.random.seed(seed_list[i_seed])
        results = concat_runs(
            results,
            simulate_random_condition(emb_true, cond_info, freeze_options)
        )
    data_rand = {'info': cond_info, 'results': results}
    return data_rand


def simulate_random_condition(emb_true, cond_info, freeze_options):
    """Simulate random selection progress for a trial configuration.

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
        # Initialize embedding.
        emb_inferred = Exponential(emb_true.n_stimuli, emb_true.n_dim)
        emb_inferred.freeze(freeze_options)
        # emb_inferred.set_log(True, delete_prev=True)  # TODO
        # Infer embedding with cold restarts.
        include_idx = np.arange(0, n_trial[i_round])
        loss[i_round] = emb_inferred.fit(
            obs.subset(include_idx), n_restart=20, init_mode='cold', verbose=0)
        # Compare the inferred model with ground truth by comparing the
        # similarity matrices implied by each model.
        simmat_infer = similarity_matrix(
            emb_inferred.similarity, emb_inferred.z['value'])
        r_squared[i_round] = matrix_correlation(simmat_infer, simmat_true)
        print(
            'Round {0} ({1} trials) | Loss: {2:.2f} | R^2: {3:.2f}'.format(
                i_round, n_trial[i_round], loss[i_round], r_squared[i_round]
            )
        )

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


def plot_exp2(results, figure_path):
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
    results_path = '/Users/bdroads/Projects/psiz-app/results'
    experiment_2(results_path)
