"""Experiment 2: Compare trial configurations and selection policies.

Since this experiment is computationally intensive, intermediate
results are saved to disk and loaded as needed.
"""

import os
import copy
import time
import itertools
from pathlib import Path

import numpy as np
from scipy.stats import sem
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from psiz.models import Exponential, load_embedding
from psiz.dimensionality import suggest_dimensionality
from psiz.simulate import Agent
from psiz.generator import RandomGenerator, ActiveGenerator
from psiz import trials
from psiz import datasets
from psiz import visualize
from psiz.utils import similarity_matrix, matrix_comparison


def experiment_2(results_path):
    """Run Experiment 2."""
    """Experiment 2a: Trial configuration comparison.

    Comparing different trial configurations using simulations.

    An initial embedding is inferred using real observations data. This
    embedding is then treated as ground truth for the purpose of
    simulating human behavior for two different display configurations.

    Notes:
        For simiplicity, the dimensionality is not inferred at each
            step. Instead, the correct dimensionality is assume to be
            known.
        Result files are saved after each run. New results are
            appeneded to existing results if the respective file already
            exists.

    """
    """Experiment 2b:  Selection comparison for one group.

    Comparing random versus active selection for one group using
    simulations. An initial embedding is inferred using real
    observations data. This embedding is then treated as ground truth
    for the purpose of simulating human behavior for two selection
    policies: random and active.

    Notes:
        For simiplicity, the dimensionality is not inferred at each
            step. Instead, the correct dimensionality is assume to be
            known.
        Result files are saved after each run. New results are
            appeneded to existing results if the respective file already
            exists.

    """
    # Settings.
    freeze_options = {'theta': {'rho': 2, 'beta': 10}}
    dataset_name = 'birds-16'
    seed_list = [913, 192, 785, 891, 841]

    # Filepaths.
    fp_emb_true = results_path / Path('emb_true_2d.hdf5')  # TODO
    fp_data_r2c1 = results_path / Path('exp_2/data_r2c1.p')
    fp_data_r8c2 = results_path / Path('exp_2/data_r8c2.p')
    fp_data_a8c2 = results_path / Path('exp_2/data_a8c2.p')
    fp_figure_embedding = results_path / Path('emb.pdf')
    fp_figure_exp2a = results_path / Path('exp_2/exp2a.pdf')
    fp_figure_exp2b = results_path / Path('exp_2/exp2b.pdf')

    # Load a set of real observations.
    (obs, catalog) = datasets.load_dataset(dataset_name)
    time_s_2c1 = 3.06  # TODO compute from obs
    time_s_8c2 = 8.98  # TODO compute from obs

    # Define experiment conditions.
    cond_info_r2c1 = {
        'name': 'Random 2-choose-1',
        'selection_policy': 'random',
        'n_reference': 2,
        'n_select': 1,
        'n_trial_initial': 500,
        'n_trial_total': 180500,
        'n_trial_per_round': 6000,
        'time_s_per_trial': time_s_2c1
    }
    cond_info_r8c2 = {
        'name': 'Random 8-choose-2',
        'selection_policy': 'random',
        'n_reference': 8,
        'n_select': 2,
        'n_trial_initial': 250,
        'n_trial_total': 15250,
        'n_trial_per_round': 500,
        'time_s_per_trial': time_s_8c2
    }
    cond_info_a8c2 = {
        'name': 'Active 8-choose-2',
        'selection_policy': 'active',
        'n_reference': 8,
        'n_select': 2,
        'n_trial_initial': 250,
        'n_trial_total': 15250,
        'n_trial_per_round': 40,
        'time_s_per_trial': time_s_8c2,
        'n_query': 10,
    }

    # Experiment 2 setup: Infer a ground-truth embedding from real
    # observations.
    # experiment_2_setup(obs, catalog, freeze_options, fp_emb_true) TODO

    emb_true = load_embedding(fp_emb_true)
    # Visualize ground-truth embedding. TODO
    # visualize.visualize_embedding_static(
    #     emb_true.z['value'], class_vec=catalog.stimuli.class_id.values,
    #     classes=catalog.class_label, filename=fp_figure_embedding)

    # simulate_multiple_runs(
    #     seed_list, emb_true, cond_info_r2c1, freeze_options, fp_data_r2c1)

    # simulate_multiple_runs(
    #     seed_list, emb_true, cond_info_r8c2, freeze_options, fp_data_r8c2)

    simulate_multiple_runs(
        seed_list, emb_true, cond_info_a8c2, freeze_options, fp_data_a8c2)

    # Visualize Experiment 2 results.
    # data_r2c1 = pickle.load(open(fp_data_r2c1, 'rb'))
    # data_r8c2 = pickle.load(open(fp_data_r8c2, 'rb'))
    # plot_exp2((data_r2c1, data_r8c2), fp_figure_exp2a)


def experiment_2_setup(obs, catalog, freeze_options, fp_emb_true):
    """Fit embedding model to real observations."""
    np.random.seed(123)

    # Determine dimensionality.
    n_dim = suggest_dimensionality(
        obs, Exponential, catalog.n_stimuli, freeze_options=freeze_options,
        verbose=1)

    # Determine embedding using all available observation data.
    emb_true = Exponential(catalog.n_stimuli, n_dim)
    emb_true.freeze(freeze_options)
    emb_true.fit(obs, n_restart=40, verbose=3)
    emb_true.save(fp_emb_true)


def simulate_multiple_runs(
        seed_list, emb_true, cond_info, freeze_options, fp_data):
    """Perform multiple runs of simulation.

    The random number generator is re-seeded before each run. Data is
    saved after each run.
    """
    if fp_data.is_file():
        data = pickle.load(open(fp_data, 'rb'))
        results = data['results']
    else:
        data = None
        results = None

    n_run = len(seed_list)
    for i_seed in range(n_run):
        np.random.seed(seed_list[i_seed])
        results_run = simulate_run(emb_true, cond_info, freeze_options, fp_data)
        results = concat_runs(results, results_run)
        data = {'info': cond_info, 'results': results}
        pickle.dump(data, open(fp_data, 'wb'))


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


def simulate_run(emb_true, cond_info, freeze_options, fp_data):
    """Simulate a single run."""
    if cond_info['selection_policy'] is 'random':
        results_run = simulate_run_random(
            emb_true, cond_info, freeze_options, fp_data)
    elif cond_info['selection_policy'] is 'active':
        results_run = simulate_run_active(
            emb_true, cond_info, freeze_options, fp_data)
    else:
        raise ValueError(
            'The `selection_policy` must be either "random" or "active".'
        )
    return results_run


def simulate_run_random(emb_true, cond_info, freeze_options, fp_data):
    """Simulate random selection progress for a trial configuration.

    Record:
        n_trial, loss, R^2
    """
    # Define agent based on true embedding.
    agent = Agent(emb_true)

    # Similarity matrix associated with true embedding.
    simmat_true = similarity_matrix(
        emb_true.similarity, emb_true.z['value'])

    # Generate a random docket of trials.
    rand_gen = RandomGenerator(
        cond_info['n_reference'], cond_info['n_select'])
    docket = rand_gen.generate(
        cond_info['n_trial_total'], emb_true.n_stimuli)
    # Simulate similarity judgments.
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

    # Initialize embedding.
    emb_inferred = Exponential(emb_true.n_stimuli, emb_true.n_dim)
    emb_inferred.freeze(freeze_options)
    for i_round in range(n_round):
        # Infer embedding.
        # if i_round < 1:
        #     init_mode = 'cold'
        # else:
        #     init_mode = 'warm'
        # emb_inferred.set_log(True, delete_prev=True)  # TODO
        init_mode = 'cold'
        include_idx = np.arange(0, n_trial[i_round])
        loss[i_round] = emb_inferred.fit(
            obs.subset(include_idx), n_restart=20, init_mode=init_mode, verbose=3)  # TODO 50, verbose
        # Compare the inferred model with ground truth by comparing the
        # similarity matrices implied by each model.
        simmat_infer = similarity_matrix(
            emb_inferred.similarity, emb_inferred.z['value'])
        r_squared[i_round] = matrix_comparison(simmat_infer, simmat_true)
        print(
            'Round {0} ({1:d} trials) | Loss: {2:.2f} | R^2: {3:.2f}'.format(
                i_round, int(n_trial[i_round]), loss[i_round], r_squared[i_round]
            )
        )
        results_temp = {
            'n_trial': np.expand_dims(n_trial[0:i_round + 1], axis=1),
            'loss': np.expand_dims(loss[0:i_round + 1], axis=1),
            'r_squared': np.expand_dims(r_squared[0:i_round + 1], axis=1)
        }
        data = {'info': cond_info, 'results': results_temp}
        pickle.dump(data, open(fp_data.absolute().as_posix() + '_temp', 'wb'))

    results = {
        'n_trial': np.expand_dims(n_trial, axis=1),
        'loss': np.expand_dims(loss, axis=1),
        'r_squared': np.expand_dims(r_squared, axis=1)
    }
    return results


def simulate_run_active(emb_true, cond_info, freeze_options, fp_data):
    """Simulate active selection progress for a trial configuration.

    Record:
        n_trial, loss, R^2
    """
    # Define agent based on true embedding.
    agent = Agent(emb_true)

    # Similarity matrix associated with true embedding.
    simmat_true = similarity_matrix(
        emb_true.similarity, emb_true.z['value'])

    # Pre-allocate metrics.
    n_round = len(np.arange(
        cond_info['n_trial_initial'],
        cond_info['n_trial_total'] + 1,
        cond_info['n_trial_per_round']
    ))
    n_trial = np.empty((n_round))
    r_squared = np.empty((n_round))
    loss = np.empty((n_round))

    # The first round is seed using a randomly generated docket.
    i_round = 0
    rand_gen = RandomGenerator(
        cond_info['n_reference'], cond_info['n_select'])
    rand_docket = rand_gen.generate(
        cond_info['n_trial_initial'], emb_true.n_stimuli)
    # Simulate similarity judgments.
    obs = agent.simulate(rand_docket)
    # Infer initial model.
    emb_inferred = Exponential(emb_true.n_stimuli, emb_true.n_dim)
    emb_inferred.freeze(freeze_options)
    n_trial[i_round] = obs.n_trial
    loss[i_round] = emb_inferred.fit(obs, n_restart=50, init_mode='cold')
    simmat_infer = similarity_matrix(
        emb_inferred.similarity, emb_inferred.z['value'])
    r_squared[i_round] = matrix_comparison(simmat_infer, simmat_true)
    print(
        'Round {0} ({1:d} trials) | Loss: {2:.2f} | R^2: {3:.2f}'.format(
            i_round, int(n_trial[i_round]), loss[i_round], r_squared[i_round]
        )
    )

    config_list = pd.DataFrame({
        'n_reference': np.array([8], dtype=np.int32),
        'n_select': np.array([2], dtype=np.int32),
        'is_ranked': [True],
        'n_outcome': np.array([56], dtype=np.int32)
    })
    active_gen = ActiveGenerator(config_list=config_list, n_neighbor=10)
    # Infer independent models with increasing amounts of data.
    for i_round in np.arange(1, n_round + 1):
        # Select trials based on expected IG.
        samples = emb_inferred.posterior_samples(obs, n_sample=2000, n_burn=100)
        time_start = time.time()
        active_docket, _ = active_gen.generate(
            cond_info['n_trial_per_round'], emb_inferred, samples,
            n_query=cond_info['n_query'])
        elapsed = time.time() - time_start
        print('Elapsed time: {0:.2f} (m)'.format(elapsed / 60))
        # Simulate observations.
        new_obs = agent.simulate(active_docket)
        obs = trials.stack([obs, new_obs])
        n_trial[i_round] = obs.n_trial

        # Infer embedding with cold restarts.
        loss[i_round] = emb_inferred.fit(obs, n_restart=50, init_mode='cold')
        # Compare the inferred model with ground truth by comparing the
        # similarity matrices implied by each model.
        simmat_infer = similarity_matrix(
            emb_inferred.similarity, emb_inferred.z['value'])
        r_squared[i_round] = matrix_comparison(simmat_infer, simmat_true)
        print(
            'Round {0} ({1:d} trials) | Loss: {2:.2f} | R^2: {3:.2f}'.format(
                i_round, int(n_trial[i_round]), loss[i_round], r_squared[i_round]
            )
        )
        results_temp = {
            'n_trial': np.expand_dims(n_trial[0:i_round + 1], axis=1),
            'loss': np.expand_dims(loss[0:i_round + 1], axis=1),
            'r_squared': np.expand_dims(r_squared[0:i_round + 1], axis=1)
        }
        data = {'info': cond_info, 'results': results_temp}
        pickle.dump(data, open(fp_data.absolute().as_posix() + '_temp', 'wb'))

    results = {
        'n_trial': np.expand_dims(n_trial, axis=1),
        'loss': np.expand_dims(loss, axis=1),
        'r_squared': np.expand_dims(r_squared, axis=1)
    }
    return results


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
        r_squared_min = np.min(r_squared, axis=1)
        r_squared_max = np.max(r_squared, axis=1)
        r_squared_sem = sem(r_squared, axis=1)

        ax.plot(
            time_cost_hr, r_squared_mean, '-', color=c_line[i_cond],
            label=name)
        ax.fill_between(
            time_cost_hr,
            r_squared_min,
            r_squared_max,
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
                time_thresh, r2_thresh + .06, "{0:.1f} hr".format(time_thresh),
                fontdict=fontdict)

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
    experiment_2(results_path)
