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
from numpy import ma
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


def main(fp_results):
    """Run all experiments.

    Experiment 0:  Ground truth embedding.
        An initial embedding is inferred using real observations data.
        This embedding is then treated as ground truth for the purpose
        of simulating human behavior in Experiment 2.

    Experiment 1: Similarity kernel comparison.

    Experiment 2a: Trial configuration comparison.
        Two different trial configurations are compared using
            simulations.

    Experiment 2b:  Selection comparison for one group.
        Random and active selection are compared using simulations.

    Notes:
        In Experiment 2, the dimensionality is not inferred at each
            step. Instead, the correct dimensionality is assumed to be
            known.
        In Experiment 2, result files are saved after each run. New
            results are appeneded to existing results if the respective
            file already exists.

    """
    # Settings.
    domain_list = ['birds']  # ['birds', 'rocks'] TODO

    for domain in domain_list:
        # Define experiment filepaths.
        fp_exp0_domain = fp_results / Path('exp_0/{0:s}'.format(domain))
        fp_exp1_domain = fp_results / Path('exp_1/{0:s}'.format(domain))
        fp_exp2_domain = fp_results / Path('exp_2/{0:s}'.format(domain))
        # Run each experiment.
        # run_exp_0(domain, fp_exp0_domain)
        # run_exp_1(domain, fp_exp0_domain, fp_exp1_domain)
        run_exp_2(domain, fp_exp0_domain, fp_exp2_domain)
    
    # Visualize Results.
    # Define filepaths.
    # fp_figure_exp2a = fp_exp2_domain / Path('exp2a.pdf')
    # fp_figure_exp2b = fp_exp2_domain / Path('exp2b.pdf')
    # data_r2c1 = pickle.load(open(fp_data_r2c1, 'rb'))
    # data_r8c2 = pickle.load(open(fp_data_r8c2, 'rb'))
    # visualize_exp_2a((data_r2c1, data_r8c2), fp_figure_exp2a)


def run_exp_0(domain, freeze_options, fp_exp0_domain):
    """Run Experiment 0."""
    # Settings.
    freeze_options = {'theta': {'rho': 2., 'tau': 1.}}
    # Define filepaths.
    fp_emb_true_2d = fp_exp0_domain / Path('emb_true_2d.hdf5')
    fp_fig_emb_true_2d = fp_exp0_domain / Path('emb_true_2d.pdf')
    fp_emb_true = fp_exp0_domain / Path('emb_true.hdf5')

    # Load the real observations.
    if domain == 'birds':
        dataset_name = 'birds-16'
    elif domain == 'rocks':
        dataset_name = 'rocks_Nosofsky_etal_2016'
    (obs, catalog) = datasets.load_dataset(dataset_name)

    np.random.seed(123)
    # Infer 2D solution for visualization purposes.
    n_dim = 2
    emb_true_2d = Exponential(catalog.n_stimuli, n_dim)
    emb_true_2d.freeze(freeze_options)
    emb_true_2d.fit(obs, n_restart=100)
    emb_true_2d.save(fp_emb_true_2d)
    # Save visualization of 2d embedding.
    visualize.visualize_embedding_static(
        emb_true_2d.z['value'], class_vec=catalog.stimuli.class_id.values,
        classes=catalog.class_label,
        filename=fp_fig_emb_true_2d.absolute().as_posix()
    )

    np.random.seed(123)
    # Determine solution by selecting dimensionality based on cross-validation.
    n_dim = suggest_dimensionality(
        obs, Exponential, catalog.n_stimuli, freeze_options=freeze_options,
        verbose=1)
    emb_true = Exponential(catalog.n_stimuli, n_dim)
    emb_true.freeze(freeze_options)
    emb_true.fit(obs, n_restart=100)
    emb_true.save(fp_emb_true)


def run_exp_2(domain, fp_exp0_domain, fp_exp2_domain):
    """Run Experiment 2."""
    # Settings.
    freeze_options = {'theta': {'rho': 2., 'tau': 1.}}
    # seed_list = [913, 192, 785, 891, 841]  # TODO
    seed_list = [192, 785, 891, 841]

    # TODO Derive from observations data.
    time_s_2c1 = 3.06
    time_s_8c2 = 8.98

    # Define experiment conditions.
    cond_info_r2c1 = {
        'name': 'Random 2-choose-1',
        'prefix': 'r2c1',
        'domain': domain,
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
        'prefix': 'r8c2',
        'domain': domain,
        'selection_policy': 'random',
        'n_reference': 8,
        'n_select': 2,
        'n_trial_initial': 500,
        'n_trial_total': 15000,
        'n_trial_per_round': 500,
        'time_s_per_trial': time_s_8c2
    }
    cond_info_a8c2 = {
        'name': 'Active 8-choose-2',
        'prefix': 'a8c2',
        'domain': domain,
        'selection_policy': 'active',
        'n_reference': 8,
        'n_select': 2,
        'n_trial_initial': 500,
        'n_trial_total': 10000,
        'n_trial_per_round': 40,
        'time_s_per_trial': time_s_8c2,
        'n_query': 40,
    }

    fp_emb_true = fp_exp0_domain / Path('emb_true.hdf5')
    emb_true = load_embedding(fp_emb_true)

    # simulate_multiple_runs(
    #     seed_list, emb_true, cond_info_r2c1, freeze_options, fp_exp2_domain)

    # simulate_multiple_runs(
    #     seed_list, emb_true, cond_info_r8c2, freeze_options, fp_exp2_domain)

    simulate_multiple_runs(
        seed_list, emb_true, cond_info_a8c2, freeze_options, fp_exp2_domain)


def simulate_multiple_runs(
        seed_list, emb_true, cond_info, freeze_options, fp_exp_domain):
    """Perform multiple runs of simulation.

    The random number generator is re-seeded before each run. Data is
    saved after each run.
    """
    fp_data = fp_exp_domain / Path('{0:s}_data.p'.format(cond_info['prefix']))
    if fp_data.is_file():
        data = pickle.load(open(fp_data, 'rb'))
        results = data['results']
    else:
        data = None
        results = None

    n_run = len(seed_list)
    for i_run in range(n_run):
        np.random.seed(seed_list[i_run])
        results_run = simulate_run(
            emb_true, cond_info, freeze_options, fp_exp_domain,
            str(seed_list[i_run])
        )
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


def simulate_run(
        emb_true, cond_info, freeze_options, fp_exp_domain, run_id):
    """Simulate a single run."""
    if cond_info['selection_policy'] is 'random':
        results_run = simulate_run_random(
            emb_true, cond_info, freeze_options, fp_exp_domain, run_id)
    elif cond_info['selection_policy'] is 'active':
        results_run = simulate_run_active(
            emb_true, cond_info, freeze_options, fp_exp_domain, run_id)
    else:
        raise ValueError(
            'The `selection_policy` must be either "random" or "active".'
        )
    return results_run


def simulate_run_random(
        emb_true, cond_info, freeze_options, fp_exp_domain, run_id):
    """Simulate random selection progress for a trial configuration.

    Record:
        n_trial, loss, R^2
    """
    # Define filepaths.
    fp_data_run = fp_exp_domain / Path('{0:s}_{1:s}_data.p'.format(cond_info['prefix'], run_id))

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
        init_mode = 'cold'
        include_idx = np.arange(0, n_trial[i_round])
        loss[i_round] = emb_inferred.fit(
            obs.subset(include_idx), n_restart=100, init_mode=init_mode)
        # Compare the inferred model with ground truth by comparing the
        # similarity matrices implied by each model.
        simmat_infer = similarity_matrix(
            emb_inferred.similarity, emb_inferred.z['value'])
        r_squared[i_round] = matrix_comparison(simmat_infer, simmat_true)
        print(
            'Round {0} ({1:d} trials) | Loss: {2:.2f} | R^2:{3:.3f} | rho: {4:.1f} | tau: {5:.1f} | beta: {6:.1f} | gamma: {7:.2g}'.format(
                i_round, int(n_trial[i_round]), loss[i_round],
                r_squared[i_round],
                emb_inferred.theta['rho']['value'],
                emb_inferred.theta['tau']['value'],
                emb_inferred.theta['beta']['value'],
                emb_inferred.theta['gamma']['value']
            )
        )
        results_temp = {
            'n_trial': np.expand_dims(n_trial[0:i_round + 1], axis=1),
            'loss': np.expand_dims(loss[0:i_round + 1], axis=1),
            'r_squared': np.expand_dims(r_squared[0:i_round + 1], axis=1)
        }
        data = {'info': cond_info, 'results': results_temp}
        pickle.dump(data, open(fp_data_run.absolute().as_posix(), 'wb'))

    results = {
        'n_trial': np.expand_dims(n_trial, axis=1),
        'loss': np.expand_dims(loss, axis=1),
        'r_squared': np.expand_dims(r_squared, axis=1)
    }
    return results


def simulate_run_active(
        emb_true, cond_info, freeze_options, fp_exp_domain, run_id):
    """Simulate active selection progress for a trial configuration.

    Record:
        n_trial, loss, R^2
    """
    # Define filepaths.
    fp_data_run = fp_exp_domain / Path('{0:s}_{1:s}_data.p'.format(cond_info['prefix'], run_id))
    fp_obs = fp_exp_domain / Path('{0:s}_{1:s}_obs.hdf5'.format(cond_info['prefix'], run_id))
    fp_emb_inf = fp_exp_domain / Path('{0:s}_{1:s}_emb_inf.hdf5'.format(cond_info['prefix'], run_id))

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
    is_valid = np.zeros((n_round), dtype=bool)
    at_criterion = False
    remaining_rounds = 10

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
    is_valid[i_round] = True
    print(
        'Round {0} ({1:d} trials) | Loss: {2:.2f} | R^2:{3:.3f} | rho: {4:.1f} | tau: {5:.1f} | beta: {6:.1f} | gamma: {7:.2g}'.format(
            i_round, int(n_trial[i_round]), loss[i_round],
            r_squared[i_round],
            emb_inferred.theta['rho']['value'],
            emb_inferred.theta['tau']['value'],
            emb_inferred.theta['beta']['value'],
            emb_inferred.theta['gamma']['value']
        )
    )
    # Freeze beta parameter.
    freeze_options = {'theta': {'beta': emb_inferred.theta['beta']['value']}}
    emb_inferred.freeze(freeze_options)

    config_list = pd.DataFrame({
        'n_reference': np.array([8], dtype=np.int32),
        'n_select': np.array([2], dtype=np.int32),
        'is_ranked': [True],
        'n_outcome': np.array([56], dtype=np.int32)
    })
    active_gen = ActiveGenerator(config_list=config_list, n_neighbor=12)
    # Infer independent models with increasing amounts of data.
    for i_round in np.arange(1, n_round + 1):
        # Select trials based on expected IG.
        time_start = time.time()
        samples = emb_inferred.posterior_samples(
            obs, n_final_sample=1000, n_burn=100, thin_step=10)
        # z_init=samples[:,:,-1]
        elapsed = time.time() - time_start
        print('Posterior | Elapsed time: {0:.2f} m'.format(elapsed / 60))
        emb_inferred.z['value'] = np.median(samples['z'], axis=2)

        time_start = time.time()
        active_docket, _ = active_gen.generate(
            cond_info['n_trial_per_round'], emb_inferred, samples,
            n_query=cond_info['n_query'])
        elapsed = time.time() - time_start
        print('Active | Elapsed time: {0:.2f} m'.format(elapsed / 60))
        # Simulate observations.
        new_obs = agent.simulate(active_docket)
        obs = trials.stack([obs, new_obs])
        n_trial[i_round] = obs.n_trial

        if np.mod(i_round, 5) == 0:
            # Infer new embedding with exact restarts.
            freeze_options = {'z': emb_inferred.z['value']}
            emb_inferred.freeze(freeze_options)
            loss[i_round] = emb_inferred.fit(
                obs, n_restart=1, init_mode='exact')
            loss[i_round] = emb_inferred.fit(
                obs, n_restart=10, init_mode='cold')
        else:
            loss[i_round] = emb_inferred.evaluate(obs)
        # Compare the inferred model with ground truth by comparing the
        # similarity matrices implied by each model.
        simmat_infer = similarity_matrix(
            emb_inferred.similarity, emb_inferred.z['value'])
        r_squared[i_round] = matrix_comparison(simmat_infer, simmat_true)
        is_valid[i_round] = True
        print(
            'Round {0} ({1:d} trials) | Loss: {2:.2f} | R^2:{3:.3f} | rho: {4:.1f} | tau: {5:.1f} | beta: {6:.1f} | gamma: {7:.2g}'.format(
                i_round, int(n_trial[i_round]), loss[i_round],
                r_squared[i_round],
                emb_inferred.theta['rho']['value'],
                emb_inferred.theta['tau']['value'],
                emb_inferred.theta['beta']['value'],
                emb_inferred.theta['gamma']['value']
            )
        )
        results_temp = {
            'n_trial': np.expand_dims(n_trial[0:i_round + 1], axis=1),
            'loss': np.expand_dims(loss[0:i_round + 1], axis=1),
            'r_squared': np.expand_dims(r_squared[0:i_round + 1], axis=1),
            'is_valid': np.expand_dims(is_valid[0:i_round + 1], axis=1)
        }
        data = {'info': cond_info, 'results': results_temp}
        pickle.dump(data, open(fp_data_run.absolute().as_posix(), 'wb'))
        obs.save(fp_obs)
        emb_inferred.save(fp_emb_inf)
        if r_squared[i_round] > .9:
            at_criterion = True
            print('Reached criterion.')

        if at_criterion:
            remaining_rounds = remaining_rounds - 1

        if remaining_rounds < 0:
            break

    results = {
        'n_trial': np.expand_dims(n_trial, axis=1),
        'loss': np.expand_dims(loss, axis=1),
        'r_squared': np.expand_dims(r_squared, axis=1),
        'is_valid': np.expand_dims(is_valid, axis=1)
    }
    return results


def visualize_exp_2(results, fp_figure):
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
    # fp_results = Path('/Users/bdroads/Projects/psiz-app/results')
    # fp_results = Path('/home/brett/packages/psiz-app/results')
    fp_results = Path('/home/brett/Projects/psiz-app.git/results')
    main(fp_results)
