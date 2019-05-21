"""Experiment 2: Compare trial configurations and selection policies.

Since this experiment is computationally intensive, intermediate
results are saved to disk and loaded as needed.
"""

import os
import copy
import time
import itertools
from pathlib import Path
import multiprocessing
from functools import partial

import numpy as np
from numpy import ma
from scipy import stats
from sklearn.model_selection import StratifiedKFold, GroupKFold
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from psiz.models import Linear, Exponential, HeavyTailed, StudentsT, load_embedding
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
    domain_list = ['birds']  # ['birds', 'rocks']

    for domain in domain_list:
        # Define experiment filepaths.
        fp_exp0_domain = fp_results / Path('exp_0/{0:s}'.format(domain))
        fp_exp1_domain = fp_results / Path('exp_1/{0:s}'.format(domain))
        fp_exp2_domain = fp_results / Path('exp_2/{0:s}'.format(domain))

        # Run each experiment.
        # run_exp_0(domain, fp_exp0_domain) TODO
        # run_exp_1(domain, fp_exp0_domain, fp_exp1_domain)
        # run_exp_2(domain, fp_exp0_domain, fp_exp2_domain) 
    fp_exp3 = fp_results / Path('exp_3')
    # run_exp_3(domain, fp_exp3)

    # Visualize Experiment 1 Results.
    # fp_cv = fp_results / Path('exp_1/{0:s}'.format(domain))
    # fp_figure_exp1 = fp_results / Path('exp_1/{0:s}/exp1.pdf'.format(domain))
    # visualize_exp_1(fp_cv, fp_figure_exp1)

    # Visualize Experiment 2 Results.
    fp_data_r2c1 = fp_results / Path('exp_2/{0:s}/r2c1/r2c1_data.p'.format(domain))
    fp_data_r8c2 = fp_results / Path('exp_2/{0:s}/r8c2/r8c2_data.p'.format(domain))
    fp_data_a8c2 = fp_results / Path('exp_2/{0:s}/a8c2/a8c2_data.p'.format(domain))
    fp_figure_exp2 = fp_results / Path('exp_2/{0:s}/exp2.pdf'.format(domain))
    data_r2c1 = pickle.load(open(fp_data_r2c1, 'rb'))
    data_r8c2 = pickle.load(open(fp_data_r8c2, 'rb'))
    data_a8c2 = pickle.load(open(fp_data_a8c2, 'rb'))
    visualize_exp_2(data_r2c1, data_r8c2, data_a8c2, fp_figure_exp2)
    # # TODO
    # fp_data_h8c2 = fp_results / Path('exp_2/{0:s}/h8c2/h8c2_913_data.p'.format(domain))
    # data_h8c2 = pickle.load(open(fp_data_h8c2, 'rb'))
    # fp_figure_exp2_plus = fp_results / Path('exp_2/{0:s}/exp2_plus.pdf'.format(domain))
    # visualize_exp_2_plus(data_r2c1, data_r8c2, data_a8c2, data_h8c2, fp_figure_exp2_plus)

    # # Visualize Experiment 3 Results.
    # fp_data_r2c1 = fp_results / Path('exp_3/r8c2_1g/r8c2_1g_data.p')
    # fp_data_r8c2 = fp_results / Path('exp_3/r8c2_2g/r8c2_2g_data.p')
    # fp_figure_exp3 = fp_results / Path('exp_3/exp3.pdf')
    # data_r8c2_g1 = pickle.load(open(fp_data_r2c1, 'rb'))
    # data_r8c2_g2 = pickle.load(open(fp_data_r8c2, 'rb'))
    # visualize_exp_3(data_r8c2_g1, data_r8c2_g2, fp_figure_exp3)


def run_exp_0(domain, fp_exp0_domain):
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
    (obs, catalog) = datasets.load_dataset(dataset_name, is_hosted=True)

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


def run_exp_1(domain, fp_exp0_domain, fp_exp1_domain):
    """Run Experiment 0."""
    # Settings.
    n_fold = 10

    # Load the real observations.
    if domain == 'birds':
        dataset_name = 'birds-16'
    elif domain == 'rocks':
        dataset_name = 'rocks_Nosofsky_etal_2016'
    (obs, catalog) = datasets.load_dataset(dataset_name, is_hosted=True)

    # Instantiate the balanced k-fold cross-validation object.
    np.random.seed(seed=4352)
    # skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=723)
    # split_list = list(skf.split(obs.stimulus_set, obs.config_idx))
    skf = GroupKFold(n_splits=n_fold)
    split_list = list(skf.split(obs.stimulus_set, obs.config_idx, obs.agent_id))
    n_fold = skf.get_n_splits()

    # Interval
    # emb = Interval(catalog.n_stimuli, n_dim=3, n_group=1)
    # freeze_options = {
    #     'theta': {
    #         'rho': 2
    #     }
    # }
    # emb.freeze(freeze_options=freeze_options)
    # emb.fit(obs, n_restart=10, verbose=3)
    # print(emb.theta)
    # visualize.visualize_embedding_static(
    #     emb.z['value'], class_vec=catalog.stimuli.class_id.values,
    #     classes=catalog.class_label,
    # )
    fp_model = fp_exp1_domain / Path('Interval')
    freeze_options = {
        'theta': {
            'rho': 2
        }
    }
    loss = embedding_cv(
        split_list, obs, Interval, catalog.n_stimuli, freeze_options)
    pickle.dump(loss, open(str(fp_model / Path("loss.p")), "wb"))

    # Exponential family.
    fp_model = fp_exp1_domain / Path('Exponential')
    freeze_options = {}
    loss = embedding_cv(
        split_list, obs, Exponential, catalog.n_stimuli, freeze_options)
    pickle.dump(loss, open(str(fp_model / Path("loss.p")), "wb"))

    # Gaussian family.
    # fp_model = fp_exp1_domain / Path('Gaussian')
    # freeze_options = {
    #     'theta': {
    #         'rho': 2,
    #         'tau': 2,
    #         'gamma': 0
    #     }
    # }
    # loss = embedding_cv(
    #     split_list, obs, Exponential, catalog.n_stimuli, freeze_options)
    # pickle.dump(loss, open(str(fp_model / Path("loss.p")), "wb"))

    # Laplacian family.
    # fp_model = fp_exp1_domain / Path('Laplacian')
    # freeze_options = {
    #     'theta': {
    #         'rho': 2,
    #         'tau': 1,
    #         'gamma': 0
    #     }
    # }
    # loss = embedding_cv(
    #     split_list, obs, Exponential, catalog.n_stimuli, freeze_options)
    # pickle.dump(loss, open(str(fp_model / Path("loss.p")), "wb"))

    # Heavy-tailed family.
    fp_model = fp_exp1_domain / Path('HeavyTailed')
    freeze_options = {}
    loss = embedding_cv(
        split_list, obs, HeavyTailed, catalog.n_stimuli, freeze_options)
    pickle.dump(loss, open(str(fp_model / Path("loss.p")), "wb"))

    # Student-t family.
    fp_model = fp_exp1_domain / Path('StudentsT')
    freeze_options = {}
    loss = embedding_cv(
        split_list, obs, StudentsT, catalog.n_stimuli, freeze_options)
    pickle.dump(loss, open(str(fp_model / Path("loss.p")), "wb"))


def run_exp_2(domain, fp_exp0_domain, fp_exp2_domain):
    """Run Experiment 2."""
    # Settings.
    freeze_options = {'theta': {'rho': 2., 'tau': 1.}}
    seed_list = [913, 192, 785, 891, 841]
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
        'n_trial_total': 150500,
        'n_trial_per_round': 5000,
        'time_s_per_trial': time_s_2c1
    }
    cond_info_r8c2 = {
        'name': 'Random 8-choose-2',
        'prefix': 'r8c2',
        'domain': domain,
        'selection_policy': 'random',
        'n_reference': 8,
        'n_select': 2,
        'n_trial_initial': 50,
        'n_trial_total': 15050,
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
        'n_trial_initial': 50,  #500,
        'n_trial_total': 10050,  #10000,
        'n_trial_per_round': 40,
        'time_s_per_trial': time_s_8c2,
        'n_query': 40,
    }
    cond_info_h8c2 = {
        'name': 'Heuristic 8-choose-2',
        'prefix': 'h8c2',
        'domain': domain,
        'selection_policy': 'heuristic',
        'n_reference': 8,
        'n_select': 2,
        'n_trial_initial': 50,
        'n_trial_total': 12050,
        'n_trial_per_round': 40,
        'time_s_per_trial': time_s_8c2,
        'n_query': 40,
    }

    fp_emb_true = fp_exp0_domain / Path('emb_true.hdf5')
    emb_true = load_embedding(fp_emb_true)

    # simulate_multiple_runs(
    #     seed_list, emb_true, cond_info_r2c1, freeze_options, fp_exp2_domain / Path(cond_info_r2c1['prefix'])
    # )

    # simulate_multiple_runs(
    #     seed_list, emb_true, cond_info_r8c2, freeze_options, fp_exp2_domain / Path(cond_info_r8c2['prefix'])
    # )

    # simulate_multiple_runs(
    #     seed_list, emb_true, cond_info_a8c2, freeze_options, fp_exp2_domain / Path(cond_info_a8c2['prefix'])
    # )

    simulate_multiple_runs(
        seed_list, emb_true, cond_info_h8c2, freeze_options, fp_exp2_domain / Path(cond_info_h8c2['prefix'])
    )


def run_exp_3(domain, fp_exp3):
    """Run Experiment 2.

    Random 8-choose-2 novices
    Random 8-choose-2 experts
    Active 8-choose-2 experts

    Generate all random novice data
    Generate all random expert data

    Infer independent expert embedding.
    Infer joint novice and expert embedding.
    """
    # Settings.
    n_stimuli = 100
    n_dim = 4
    n_group = 2
    freeze_options = {'theta': {'rho': 2., 'tau': 1.}}
    seed_list = [913, 192, 785, 891, 841]
    
    # Define experiment conditions.
    cond_info_r8c2_1g = {
        'name': 'Random 8-choose-2',
        'prefix': 'r8c2_1g',
        'domain': domain,
        'selection_policy': 'random',
        'n_reference': 8,
        'n_select': 2,
        'n_trial_initial': 50,
        'n_trial_total': 8050,
        'n_trial_per_round': 250,
    }

    cond_info_r8c2_2g = {
        'name': 'Random 8-choose-2',
        'prefix': 'r8c2_2g',
        'domain': domain,
        'selection_policy': 'random-existing',
        'n_reference': 8,
        'n_select': 2,
        'n_trial_initial': 50,
        'n_trial_total': 3550,
        'n_trial_per_round': 250,
    }

    # Generate ground-truth embedding.
    np.random.seed(548)
    emb_true = exp_3_ground_truth(n_stimuli, n_dim, n_group)

    # Generate a random docket of trials to show novices for each run.
    generator = RandomGenerator(
        cond_info_r8c2_2g['n_reference'],
        cond_info_r8c2_2g['n_select'])
    docket = generator.generate(
        cond_info_r8c2_2g['n_trial_total'], emb_true.n_stimuli
    )

    # Simulate novice similarity judgments.
    agent_novice = Agent(emb_true, group_id=0)
    obs_novice = agent_novice.simulate(docket)

    # TODO
    # simulate_multiple_runs(
    #     seed_list, emb_true, cond_info_r8c2_1g, freeze_options,
    #     fp_exp3 / Path(cond_info_r8c2_1g['prefix']), group_id=1
    # )

    simulate_multiple_runs(
        seed_list, emb_true, cond_info_r8c2_2g, freeze_options,
        fp_exp3 / Path(cond_info_r8c2_2g['prefix']), group_id=1,
        obs_existing=obs_novice
    )


def embedding_cv(
        split_list, obs, embedding_constructor, n_stimuli, freeze_options):
    """Embedding cross-validation procedure."""
    # Cross-validation settings.
    verbose = 2
    n_fold = len(split_list)
    J_train = np.empty((n_fold))
    J_test = np.empty((n_fold))

    loaded_fold = partial(
        evaluate_fold, split_list=split_list, obs=obs,
        embedding_constructor=embedding_constructor, n_stimuli=n_stimuli,
        freeze_options=freeze_options, verbose=verbose)

    fold_list = range(n_fold)
    # for i_fold in fold_list:
    #     (J_train[i_fold], J_test[i_fold]) = loaded_func(i_fold)
    with multiprocessing.Pool() as pool:
        results = pool.map(loaded_fold, fold_list)

    J_train = []
    J_test = []
    n_dim = []
    for i_fold in fold_list:
        J_train.append(results[i_fold][0])
        J_test.append(results[i_fold][1])
        n_dim.append(results[i_fold][2])

    return {'train': J_train, 'test': J_test, 'n_dim': n_dim}


def evaluate_fold(
        i_fold, split_list, obs, embedding_constructor, n_stimuli,
        freeze_options, verbose):
    """Evaluate fold."""
    # Settings.
    n_restart_dim = 20
    n_restart_fit = 50

    if verbose > 1:
        print('    Fold: ', i_fold)

    (train_index, test_index) = split_list[i_fold]

    n_group = len(np.unique(obs.group_id))

    # Train.
    obs_train = obs.subset(train_index)
    # Select dimensionality.
    n_dim = suggest_dimensionality(
        obs_train, embedding_constructor, n_stimuli, n_restart=n_restart_dim,
        freeze_options=freeze_options
    )
    if verbose > 1:
        print("        Suggested dimensionality: {0}".format(n_dim))

    # Instantiate model.
    embedding_model = embedding_constructor(n_stimuli, n_dim, n_group)
    if len(freeze_options) > 0:
        embedding_model.freeze(freeze_options=freeze_options)
    # Fit model using training data.
    J_train = embedding_model.fit(obs_train, n_restart=n_restart_fit)

    # Test.
    obs_test = obs.subset(test_index)
    J_test = embedding_model.evaluate(obs_test)

    return (J_train, J_test, n_dim)


def simulate_multiple_runs(
        seed_list, emb_true, cond_info, freeze_options, dir_cond,
        group_id=0, obs_existing=None):
    """Perform multiple runs of simulation.

    The random number generator is re-seeded before each run. Data is
    saved after each run.
    """
    fp_data = dir_cond / Path('{0:s}_data.p'.format(cond_info['prefix']))
    if fp_data.is_file():
        data = pickle.load(open(fp_data, 'rb'))
        results = data['results']
    else:
        # Make directory
        if not os.path.exists(dir_cond):
            os.makedirs(dir_cond)
        data = None
        results = None

    n_run = len(seed_list)
    for i_run in range(n_run):
        np.random.seed(seed_list[i_run])
        results_run = simulate_run(
            emb_true, cond_info, freeze_options, dir_cond,
            str(seed_list[i_run]), group_id=group_id, obs_existing=obs_existing
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
        emb_true, cond_info, freeze_options, dir_cond, run_id,
        group_id=0, obs_existing=None):
    """Simulate a single run."""
    if cond_info['selection_policy'] == 'random':
        results_run = simulate_run_random(
            emb_true, cond_info, freeze_options, dir_cond, run_id,
            group_id=group_id
        )
    elif cond_info['selection_policy'] == 'active':
        results_run = simulate_run_active(
            emb_true, cond_info, freeze_options, dir_cond, run_id,
            group_id=group_id
        )
    elif cond_info['selection_policy'] == 'random-existing':
        results_run = simulate_run_random_existing(
            emb_true, cond_info, freeze_options, dir_cond, run_id,
            group_id, obs_existing
        )
    elif cond_info['selection_policy'] == 'heuristic':
        results_run = simulate_run_hueristic(
            emb_true, cond_info, freeze_options, dir_cond, run_id,
            group_id=group_id
        )
    else:
        raise ValueError(
            'The `selection_policy` {0} is not defined. Must be either '
            '"random", "random-existing", or "active".'.format(
                cond_info['selection_policy']
            )
        )
    return results_run


def simulate_run_random(
        emb_true, cond_info, freeze_options, dir_cond, run_id, group_id=0):
    """Simulate random selection progress for a trial configuration.

    Record:
        n_trial, loss, r^2
    """
    # Define filepaths.
    fp_data_run = dir_cond / Path('{0:s}_{1:s}_data.p'.format(cond_info['prefix'], run_id))
    fp_obs = dir_cond / Path('{0:s}_{1:s}_obs.hdf5'.format(cond_info['prefix'], run_id))
    fp_emb_inf = dir_cond / Path('{0:s}_{1:s}_emb_inf.hdf5'.format(cond_info['prefix'], run_id))

    # Define agent based on true embedding.
    agent = Agent(emb_true, group_id=group_id)

    # Similarity matrix associated with true embedding.
    def sim_func_true(z_q, z_ref):
        return emb_true.similarity(z_q, z_ref, group_id=group_id)
    simmat_true = similarity_matrix(
        sim_func_true, emb_true.z['value']
    )

    # Generate a random docket of trials.
    rand_gen = RandomGenerator(
        cond_info['n_reference'], cond_info['n_select'])
    docket = rand_gen.generate(
        cond_info['n_trial_total'], emb_true.n_stimuli)
    # Simulate similarity judgments.
    obs = agent.simulate(docket)
    obs.set_group_id(0)

    # Infer independent models with increasing amounts of data.
    n_trial = np.arange(
        cond_info['n_trial_initial'],
        cond_info['n_trial_total'] + 1,
        cond_info['n_trial_per_round']
    )
    n_round = len(n_trial)
    r_squared = np.empty((n_round))
    loss = np.empty((n_round))
    is_valid = np.zeros((n_round), dtype=bool)

    # Initialize embedding.
    emb_inferred = Exponential(
        emb_true.n_stimuli, n_dim=emb_true.n_dim, n_group=1
    )
    emb_inferred.freeze(freeze_options)
    for i_round in range(n_round):
        # Infer embedding.
        init_mode = 'cold'
        include_idx = np.arange(0, n_trial[i_round])
        loss[i_round] = emb_inferred.fit(
            obs.subset(include_idx), n_restart=50, init_mode=init_mode)
        # Compare the inferred model with ground truth by comparing the
        # similarity matrices implied by each model.
        simmat_infer = similarity_matrix(
            emb_inferred.similarity, emb_inferred.z['value'])
        r_squared[i_round] = matrix_comparison(simmat_infer, simmat_true)
        is_valid[i_round] = True
        print(
            'Round {0} ({1:d} trials) | Loss: {2:.2f} | r^2:{3:.3f} | rho: {4:.1f} | tau: {5:.1f} | beta: {6:.1f} | gamma: {7:.2g}'.format(
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
        # emb_inferred.save(fp_emb_inf)

    results = {
        'n_trial': np.expand_dims(n_trial, axis=1),
        'loss': np.expand_dims(loss, axis=1),
        'r_squared': np.expand_dims(r_squared, axis=1),
        'is_valid': np.expand_dims(is_valid, axis=1)
    }
    return results


def simulate_run_active(
        emb_true, cond_info, freeze_options, dir_cond, run_id, group_id=0):
    """Simulate active selection progress for a trial configuration.

    Record:
        n_trial, loss, r^2
    """
    # Define filepaths.
    fp_data_run = dir_cond / Path('{0:s}_{1:s}_data.p'.format(cond_info['prefix'], run_id))
    fp_obs = dir_cond / Path('{0:s}_{1:s}_obs.hdf5'.format(cond_info['prefix'], run_id))
    fp_emb_inf = dir_cond / Path('{0:s}_{1:s}_emb_inf.hdf5'.format(cond_info['prefix'], run_id))

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
        'Round {0} ({1:d} trials) | Loss: {2:.2f} | r^2:{3:.3f} | rho: {4:.1f} | tau: {5:.1f} | beta: {6:.1f} | gamma: {7:.2g}'.format(
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

        # if (i_round < 5) or (np.mod(i_round, 5) == 0):  # TODO try
        if np.mod(i_round, 5) == 0:
            # Infer new embedding with exact restarts.
            freeze_options = {'z': emb_inferred.z['value']}
            emb_inferred.freeze(freeze_options)
            loss[i_round] = emb_inferred.fit(
                obs, n_restart=3, init_mode='exact')
            emb_inferred.thaw(['z'])
            loss[i_round] = emb_inferred.fit(
                obs, n_restart=50, init_mode='cold')
        else:
            loss[i_round] = emb_inferred.evaluate(obs)
        # Compare the inferred model with ground truth by comparing the
        # similarity matrices implied by each model.
        simmat_infer = similarity_matrix(
            emb_inferred.similarity, emb_inferred.z['value'])
        r_squared[i_round] = matrix_comparison(simmat_infer, simmat_true)
        is_valid[i_round] = True
        print(
            'Round {0} ({1:d} trials) | Loss: {2:.2f} | r^2:{3:.3f} | rho: {4:.1f} | tau: {5:.1f} | beta: {6:.1f} | gamma: {7:.2g}'.format(
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
        if r_squared[i_round] > .91:
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


def simulate_run_hueristic(
        emb_true, cond_info, freeze_options, dir_cond, run_id, group_id=0):
    """Simulate heuristic selection progress for a trial configuration.

    Record:
        n_trial, loss, r^2
    """
    # Define filepaths.
    fp_data_run = dir_cond / Path('{0:s}_{1:s}_data.p'.format(cond_info['prefix'], run_id))
    fp_obs = dir_cond / Path('{0:s}_{1:s}_obs.hdf5'.format(cond_info['prefix'], run_id))
    fp_emb_inf = dir_cond / Path('{0:s}_{1:s}_emb_inf.hdf5'.format(cond_info['prefix'], run_id))

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
        'Round {0} ({1:d} trials) | Loss: {2:.2f} | r^2:{3:.3f} | rho: {4:.1f} | tau: {5:.1f} | beta: {6:.1f} | gamma: {7:.2g}'.format(
            i_round, int(n_trial[i_round]), loss[i_round],
            r_squared[i_round],
            emb_inferred.theta['rho']['value'],
            emb_inferred.theta['tau']['value'],
            emb_inferred.theta['beta']['value'],
            emb_inferred.theta['gamma']['value']
        )
    )

    config_list = pd.DataFrame({
        'n_reference': np.array([8], dtype=np.int32),
        'n_select': np.array([2], dtype=np.int32),
        'is_ranked': [True],
        'n_outcome': np.array([56], dtype=np.int32)
    })
    heuristic_gen = HeuristicGenerator(config_list=config_list)
    # Infer independent models with increasing amounts of data.
    for i_round in np.arange(1, n_round + 1):
        # Select next docket.
        active_docket = heuristic_gen.generate(
            cond_info['n_trial_per_round'], emb_inferred, obs=obs
        )
        # Simulate observations.
        new_obs = agent.simulate(active_docket)
        obs = trials.stack([obs, new_obs])
        n_trial[i_round] = obs.n_trial

        # Infer new embedding.
        time_start = time.time()
        loss[i_round] = emb_inferred.fit(obs, n_restart=50)
        elapsed = time.time() - time_start
        print('Inference | Elapsed time: {0:.2f} m'.format(elapsed / 60))

        # Compare the inferred model with ground truth by comparing the
        # similarity matrices implied by each model.
        simmat_infer = similarity_matrix(
            emb_inferred.similarity, emb_inferred.z['value'])
        r_squared[i_round] = matrix_comparison(simmat_infer, simmat_true)
        is_valid[i_round] = True
        print(
            'Round {0} ({1:d} trials) | Loss: {2:.2f} | r^2:{3:.3f} | rho: {4:.1f} | tau: {5:.1f} | beta: {6:.1f} | gamma: {7:.2g}'.format(
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
        if r_squared[i_round] > .91:
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


def exp_3_ground_truth(n_stimuli, n_dim, n_group):
    """Return a ground truth embedding for experiment 3."""
    emb = Exponential(
        n_stimuli, n_dim=n_dim, n_group=n_group)
    mean = np.zeros((n_dim))
    cov = .03 * np.identity(n_dim)
    z = np.random.multivariate_normal(mean, cov, (n_stimuli))
    attention = np.array((
        (1.8, 1.8, .2, .2),
        (.2, .2, 1.8, 1.8)
    ))
    freeze_options = {
        'z': z,
        'theta': {
            'rho': 2,
            'tau': 1,
            'beta': 10,
            'gamma': 0.001
        },
        'phi': {
            'phi_1': attention
        }
    }
    emb.freeze(freeze_options)
    # sim_mat = similarity_matrix(emb.similarity, z)
    # idx_upper = np.triu_indices(n_stimuli, 1)
    # plt.hist(sim_mat[idx_upper])
    # plt.show()
    return emb


def simulate_run_random_existing(
        emb_true, cond_info, freeze_options, dir_cond, run_id,
        group_id, obs_existing):
    """Simulate random selection progress for a trial configuration.

    Record:
        n_trial, loss, r^2
    """
    # Define filepaths.
    fp_data_run = dir_cond / Path('{0:s}_{1:s}_data.p'.format(cond_info['prefix'], run_id))
    fp_obs = dir_cond / Path('{0:s}_{1:s}_obs.hdf5'.format(cond_info['prefix'], run_id))
    fp_emb_inf = dir_cond / Path('{0:s}_{1:s}_emb_inf.hdf5'.format(cond_info['prefix'], run_id))

    # Define expert agent based on true embedding.
    agent_expert = Agent(emb_true, group_id=1)

    def sim_func_true_novice(z_q, z_ref):
        return emb_true.similarity(z_q, z_ref, group_id=0)
    simmat_novice_true = similarity_matrix(
        sim_func_true_novice, emb_true.z['value']
    )

    def sim_func_true_expert(z_q, z_ref):
        return emb_true.similarity(z_q, z_ref, group_id=1)
    simmat_expert_true = similarity_matrix(
        sim_func_true_expert, emb_true.z['value']
    )

    # Generate a random docket of trials.
    rand_gen = RandomGenerator(
        cond_info['n_reference'], cond_info['n_select'])
    docket = rand_gen.generate(
        cond_info['n_trial_total'], emb_true.n_stimuli)
    # Simulate expert similarity judgments.
    obs_expert = agent_expert.simulate(docket)
    obs_expert.save(fp_obs)

    # Infer independent models with increasing amounts of data.
    n_trial = np.arange(
        cond_info['n_trial_initial'],
        cond_info['n_trial_total'] + 1,
        cond_info['n_trial_per_round']
    )
    n_round = len(n_trial)
    r_squared = np.empty((n_round))
    loss = np.empty((n_round))
    is_valid = np.zeros((n_round), dtype=bool)

    # Initialize embedding.
    emb_inferred = Exponential(
        emb_true.n_stimuli, emb_true.n_dim, emb_true.n_group
    )
    emb_inferred.freeze(freeze_options)
    for i_round in range(n_round):
        # Infer embedding.
        init_mode = 'cold'
        include_idx = np.arange(0, n_trial[i_round])

        obs_round = trials.stack(
            (obs_existing, obs_expert.subset(include_idx))
        )

        loss[i_round] = emb_inferred.fit(
            obs_round, n_restart=50, init_mode=init_mode
        )
        # Compare the inferred model with ground truth by comparing the
        # similarity matrices implied by each model.

        def sim_func_infer_novice(z_q, z_ref):
            return emb_inferred.similarity(z_q, z_ref, group_id=0)
        simmat_novice_infer = similarity_matrix(
            sim_func_infer_novice, emb_inferred.z['value']
        )

        def sim_func_infer_expert(z_q, z_ref):
            return emb_inferred.similarity(z_q, z_ref, group_id=1)
        simmat_expert_infer = similarity_matrix(
            sim_func_infer_expert, emb_inferred.z['value']
        )

        r_pearson_2 = np.zeros([2, 2], dtype=np.float)
        r_pearson_2[0, 0] = matrix_comparison(
            simmat_novice_infer, simmat_novice_true, score='pearson'
        )
        r_pearson_2[0, 1] = matrix_comparison(
            simmat_novice_infer, simmat_expert_true, score='pearson'
        )
        r_pearson_2[1, 0] = matrix_comparison(
            simmat_expert_infer, simmat_novice_true, score='pearson'
        )
        r_pearson_2[1, 1] = matrix_comparison(
            simmat_expert_infer, simmat_expert_true, score='pearson'
        )

        r_squared[i_round] = matrix_comparison(
            simmat_expert_infer, simmat_expert_true,
        )
        is_valid[i_round] = True
        print(
            'Round {0} ({1:d} trials) | Loss: {2:.2f} | r^2:{3:.3f} | rho: {4:.1f} | tau: {5:.1f} | beta: {6:.1f} | gamma: {7:.2g}'.format(
                i_round, int(n_trial[i_round]), loss[i_round],
                r_squared[i_round],
                emb_inferred.theta['rho']['value'],
                emb_inferred.theta['tau']['value'],
                emb_inferred.theta['beta']['value'],
                emb_inferred.theta['gamma']['value']
            )
        )
        print('    ============================')
        print('      True  |    Inferred')
        print('            | Novice  Expert')
        print('    --------+-------------------')
        print('     Novice | {0: >6.2f}  {1: >6.2f}'.format(
            r_pearson_2[0, 0], r_pearson_2[0, 1]))
        print('     Expert | {0: >6.2f}  {1: >6.2f}'.format(
            r_pearson_2[1, 0], r_pearson_2[1, 1]))
        # print('\n')

        results_temp = {
            'n_trial': np.expand_dims(n_trial[0:i_round + 1], axis=1),
            'loss': np.expand_dims(loss[0:i_round + 1], axis=1),
            'r_squared': np.expand_dims(r_squared[0:i_round + 1], axis=1),
            'is_valid': np.expand_dims(is_valid[0:i_round + 1], axis=1)
        }
        data = {'info': cond_info, 'results': results_temp}
        pickle.dump(data, open(fp_data_run.absolute().as_posix(), 'wb'))

    results = {
        'n_trial': np.expand_dims(n_trial, axis=1),
        'loss': np.expand_dims(loss, axis=1),
        'r_squared': np.expand_dims(r_squared, axis=1),
        'is_valid': np.expand_dims(is_valid, axis=1)
    }
    return results


def visualize_exp_1(fp_cv, fp_figure=None):
    """Plot results."""
    # 'Gaussian', 'Laplacian', 'StudentsT'
    model_list = ['Linear', 'Exponential', 'HeavyTailed', 'StudentsT']
    pretty_list = ['Linear', 'Exponential', 'Heavy-Tailed', 'Student-t']
    n_model = len(model_list)

    train_mu = np.empty(n_model)
    train_se = np.empty(n_model)
    test_mu = np.empty(n_model)
    test_se = np.empty(n_model)
    test_sd = np.empty(n_model)
    loss_all = []
    for i_model, model_name in enumerate(model_list):
        filepath = fp_cv / Path(model_name, 'loss.p')
        loss = pickle.load(open(str(filepath), "rb"))
        loss_all.append(loss)

        train_mu[i_model] = np.mean(loss['train'])
        train_se[i_model] = stats.sem(loss['train'])

        test_mu[i_model] = np.mean(loss['test'])
        test_se[i_model] = stats.sem(loss['test'])
        test_sd[i_model] = np.std(loss['test'])
        print(
            '{0:s} | M={1:.2f}, SD={2:.2f}'.format(
                model_name, test_mu[i_model], test_sd[i_model]
            )
        )

    print_ttest(
        model_list[0], model_list[1],
        loss_all[0]['test'], loss_all[1]['test']
    )
    print_ttest(
        model_list[0], model_list[2],
        loss_all[0]['test'], loss_all[2]['test']
    )
    print_ttest(
        model_list[1], model_list[2],
        loss_all[1]['test'], loss_all[2]['test']
    )

    ind = np.arange(n_model)

    # Determine the maximum and minimum y values in the figure.
    ymin = np.min(np.stack((train_mu - train_se, test_mu - test_se), axis=0))
    ymax = np.max(np.stack((train_mu + train_se, test_mu + test_se), axis=0))
    ydiff = ymax - ymin
    ypad = ydiff * .1

    # width = 0.35       # the width of the bars
    width = 0.75

    fig, ax = plt.subplots()
    # rects1 = ax.bar(ind, train_mu, width, color='r', yerr=train_se)
    # rects2 = ax.bar(ind + width, test_mu, width, color='b', yerr=test_se)
    error_kw = {'linewidth': 2, 'ecolor': 'r'}
    rects2 = ax.bar(ind, test_mu, width, color='b', yerr=test_se, error_kw=error_kw)

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Loss')
    ax.set_title('Average Validation Loss')
    ax.set_xticks(ind)
    # ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(pretty_list)

    # ax.legend((rects1[0], rects2[0]), ('Train', 'Test'), loc=4)
    axes = plt.gca()
    # axes.set_xlim([xmin,xmax])
    axes.set_ylim([ymin - ypad, ymax + ypad])

    if fp_figure is None:
        plt.show()
    else:
        plt.savefig(fp_figure.absolute().as_posix(), format='pdf', bbox_inches="tight", dpi=100)


def print_ttest(m1, m2, x1, x2):
    """Print t-test results in APA format."""
    [t, p] = stats.ttest_ind(x1, x2)
    df = len(x1) - 1
    print(
        '{0} | M = {1:.2f}, SD = {2:.2f}'.format(
            m1, np.mean(x1), np.std(x1)
        )
    )
    print(
        '{0} | M = {1:.2f}, SD = {2:.2f}'.format(
            m2, np.mean(x2), np.std(x2)
        )
    )
    print(
        '{0}:{1} | t({2}) = {3:.2f}, p = {4:.2f}'.format(
            m1, m2, df, t, p
        )
    )


def visualize_exp_2(data_r2c1, data_r8c2, data_a8c2, fp_figure=None):
    """Visualize results of experiment."""
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
        'verticalalignment': 'center',
        'horizontalalignment': 'center'
    }

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    i_cond = 2
    plot_exp2_condition(
        ax, data_r2c1, c_line[i_cond], c_env[i_cond], c_scatter[i_cond],
        fontdict
    )

    i_cond = 0
    plot_exp2_condition(
        ax, data_r8c2, c_line[i_cond], c_env[i_cond], c_scatter[i_cond],
        fontdict
    )

    i_cond = 1
    plot_exp2_condition(
        ax, data_a8c2, c_line[i_cond], c_env[i_cond], c_scatter[i_cond],
        fontdict, thin_step=5
    )

    ax.set_ylim(bottom=0., top=1.05)
    ax.set_xlabel('Total Worker Hours')
    ax.set_ylabel(r'Pearson $\rho$')
    ax.legend()
    plt.tight_layout()

    if fp_figure is None:
        plt.show()
    else:
        plt.savefig(
            fp_figure.absolute().as_posix(), format='pdf',
            bbox_inches="tight", dpi=100)


def visualize_exp_2_plus(data_r2c1, data_r8c2, data_a8c2, data_h8c2, fp_figure=None):
    """Visualize results of experiment."""
    # Standard.
    rgb1 = np.array((0.0, 0.0, 0.5312, 1.0))
    rgb2 = np.array((1.0, 0.8125, 0.0, 1.0))
    rgb3 = np.array((0.5, 0.0, 0.0, 1.0))
    rgb4 = np.array([0. , 0.6, 0.2, 1.0])
    # Transparent version.
    alpha_val = .2
    rgb1_trans = np.array((0.0, 0.0, 0.5312, alpha_val))
    rgb2_trans = np.array((1.0, 0.8125, 0.0, alpha_val))
    rgb3_trans = np.array((0.5, 0.0, 0.0, alpha_val))
    rgb4_trans = np.array([0. , 0.6, 0.2, alpha_val])
    # Lighter version.
    color_scale = .4  # Lower scale yeilds lighter colors.
    rgb1_light = 1 - (color_scale * (1 - rgb1))
    rgb2_light = 1 - (color_scale * (1 - rgb2))
    rgb3_light = 1 - (color_scale * (1 - rgb3))
    rgb4_light = 1 - (color_scale * (1 - rgb4))

    c_line = [tuple(rgb1), tuple(rgb3), tuple(rgb2), tuple(rgb4)]
    # c_env = [tuple(rgb1_light), tuple(rgb3_light)]
    c_env = [tuple(rgb1_trans), tuple(rgb3_trans), tuple(rgb2_trans), tuple(rgb4_trans)]
    c_scatter = [
        np.expand_dims(rgb1, axis=0),
        np.expand_dims(rgb3, axis=0),
        np.expand_dims(rgb2, axis=0),
        np.expand_dims(rgb4, axis=0)
    ]

    fontdict = {
        'fontsize': 10,
        'verticalalignment': 'top',
        'horizontalalignment': 'left'
    }

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    i_cond = 2
    plot_exp2_condition(
        ax, data_r2c1, c_line[i_cond], c_env[i_cond], c_scatter[i_cond],
        fontdict
    )

    i_cond = 0
    plot_exp2_condition(
        ax, data_r8c2, c_line[i_cond], c_env[i_cond], c_scatter[i_cond],
        fontdict
    )

    i_cond = 1
    plot_exp2_condition(
        ax, data_a8c2, c_line[i_cond], c_env[i_cond], c_scatter[i_cond],
        fontdict, thin_step=5
    )

    i_cond = 3
    plot_exp2_condition(
        ax, data_h8c2, c_line[i_cond], c_env[i_cond], c_scatter[i_cond],
        fontdict, thin_step=5
    )

    ax.set_ylim(bottom=0., top=1.05)
    ax.set_xlabel('Total Worker Hours')
    ax.set_ylabel(r'Pearson $\rho$')
    ax.legend()
    plt.tight_layout()

    if fp_figure is None:
        plt.show()
    else:
        plt.savefig(
            fp_figure.absolute().as_posix(), format='pdf',
            bbox_inches="tight", dpi=100)


def plot_exp2_condition(
        ax, data, c_line, c_env, c_scatter, fontdict, rsquared_crit=.95, thin_step=1):
    """Plot condition."""
    legend_name = data['info']['name']
    results = data['results']

    if 'is_valid' in results:
        results["is_valid"] = results["is_valid"][0::thin_step, :]
    results["loss"] = results["loss"][0::thin_step, :]
    results["n_trial"] = results["n_trial"][0::thin_step, :]
    results["r_squared"] = results["r_squared"][0::thin_step, :]

    time_factor = data['info']['time_s_per_trial'] / (60 * 60)
    n_run = results['n_trial'].shape[1]

    if 'is_valid' in results:
        is_valid = results['is_valid']
    else:
        is_valid = np.ones(results['r_squared'].shape, dtype=bool)

    n_trial_mask = ma.array(results['n_trial'], mask=np.logical_not(is_valid))
    n_trial_avg = np.mean(n_trial_mask, axis=1)

    r_mask = ma.array(results['r_squared'], mask=np.logical_not(is_valid))**.5

    r_mean_avg = np.mean(r_mask, axis=1)
    r_mean_min = np.min(r_mask, axis=1)
    r_mean_max = np.max(r_mask, axis=1)

    xg = np.log10(time_factor * n_trial_avg)
    ax.plot(
        xg, r_mean_avg, '-', color=c_line,
        label='{0:s}'.format(legend_name)
    )
    ax.fill_between(
        xg, r_mean_min, r_mean_max,
        facecolor=c_env, edgecolor='none'
    )

    # Plot Criterion
    dmy_idx = np.arange(len(r_mean_avg))
    locs = np.greater_equal(r_mean_avg, rsquared_crit)
    if np.sum(locs) > 0:
        after_idx = dmy_idx[locs]
        after_idx = after_idx[0]
        before_idx = after_idx - 1
        segment_rise = r_mean_avg[after_idx] - r_mean_avg[before_idx]
        segment_run = n_trial_avg[after_idx] - n_trial_avg[before_idx]
        segment_slope = segment_rise / segment_run
        xg = np.arange(segment_run, dtype=np.int)
        yg = segment_slope * xg + r_mean_avg[before_idx]

        locs2 = np.greater_equal(yg, rsquared_crit)
        trial_thresh = xg[locs2]
        trial_thresh = trial_thresh[0]
        r2_thresh = yg[trial_thresh]
        trial_thresh = trial_thresh + n_trial_avg[before_idx]

        ax.scatter(
            np.log10(time_factor * trial_thresh), r2_thresh, marker='d', color=c_scatter,
            edgecolors='k'
        )
        ax.text(
            np.log10(time_factor * trial_thresh), r2_thresh + .04,
            "{0:.1f}".format(time_factor * trial_thresh), fontdict=fontdict
        )
    # Major ticks.
    ax.set_xticks([-1, 0, 1, 2])
    ax.set_xticklabels([".1", "1", "10", "100"])
    # Minor ticks.
    lg = np.hstack((np.arange(.1, 1, .1), np.arange(1, 10, 1), np.arange(10, 100, 10)))
    ax.set_xticks(np.log10(lg), minor=True)


def visualize_exp_3(data_r8c2_g1, data_r8c2_g2, fp_figure=None):
    """Visualize experiment 3."""
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

    i_cond = 0
    plot_exp3_condition(
        ax, data_r8c2_g1, c_line[i_cond], c_env[i_cond], c_scatter[i_cond],
        fontdict, 'Independent'
    )

    i_cond = 1
    plot_exp3_condition(
        ax, data_r8c2_g2, c_line[i_cond], c_env[i_cond], c_scatter[i_cond],
        fontdict, 'Shared'
    )

    ax.set_ylim(bottom=0., top=1.05)
    ax.set_xlabel('Total Expert Worker Hours')
    ax.set_ylabel(r'Pearson $\rho$')
    ax.legend()
    plt.tight_layout()

    if fp_figure is None:
        plt.show()
    else:
        plt.savefig(
            fp_figure.absolute().as_posix(), format='pdf',
            bbox_inches="tight", dpi=100)


def plot_exp3_condition(
        ax, data, c_line, c_env, c_scatter, fontdict, legend_name,
        rsquared_crit=.95):
    """Plot condition."""
    results = data['results']

    time_s_8c2 = 8.98
    time_factor = time_s_8c2 / (60 * 60)
    n_run = results['n_trial'].shape[1]

    if 'is_valid' in results:
        is_valid = results['is_valid']
    else:
        is_valid = np.ones(results['r_squared'].shape, dtype=bool)

    n_trial_mask = ma.array(results['n_trial'], mask=np.logical_not(is_valid))
    n_trial_avg = np.mean(n_trial_mask, axis=1)
    
    r_mask = ma.array(results['r_squared'], mask=np.logical_not(is_valid))**.5
    
    r_mean_avg = np.mean(r_mask, axis=1)
    r_mean_min = np.min(r_mask, axis=1)
    r_mean_max = np.max(r_mask, axis=1)

    ax.semilogx(
        time_factor * n_trial_avg, r_mean_avg, '-', color=c_line,
        label='{0:s}'.format(legend_name)
        # label='{0:s} ({1:d})'.format(legend_name, n_run)
    )
    ax.fill_between(
        time_factor * n_trial_avg, r_mean_min, r_mean_max,
        facecolor=c_env, edgecolor='none'
    )

    # Plot Criterion
    dmy_idx = np.arange(len(r_mean_avg))
    locs = np.greater_equal(r_mean_avg, rsquared_crit)
    if np.sum(locs) > 0:
        after_idx = dmy_idx[locs]
        after_idx = after_idx[0]
        before_idx = after_idx - 1
        segment_rise = r_mean_avg[after_idx] - r_mean_avg[before_idx]
        segment_run = n_trial_avg[after_idx] - n_trial_avg[before_idx]
        segment_slope = segment_rise / segment_run
        xg = np.arange(segment_run, dtype=np.int)
        yg = segment_slope * xg + r_mean_avg[before_idx]

        locs2 = np.greater_equal(yg, rsquared_crit)
        trial_thresh = xg[locs2]
        trial_thresh = trial_thresh[0]
        r2_thresh = yg[trial_thresh]
        trial_thresh = trial_thresh + n_trial_avg[before_idx]

        ax.scatter(
            time_factor * trial_thresh, r2_thresh, marker='d', color=c_scatter,
            edgecolors='k')
        ax.text(
            time_factor * trial_thresh, r2_thresh + .06, "{0:.1f}".format(time_factor * trial_thresh),
            fontdict=fontdict)


if __name__ == "__main__":
    # Specify the path to a folder where you would like to store all your
    # results by change the following line. For example,
    # fp_results = Path('/home/results').
    # fp_results = Path('/home/brett/projects/psiz-app/results')
    fp_results = Path('/Users/bdroads/Projects/psiz-app/results')
    main(fp_results)
