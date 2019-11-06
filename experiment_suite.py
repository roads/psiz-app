"""Experiment Suite.

This script carries out the experiments described in:

Roads, B.D. & Mozer, M. C. (2019). Obtaining psychological embeddings
    through joint kernel and metric learning. Behavior Research Methods.
    doi: 10.3758/s13428-019-01275-5

Since this experiment is computationally intensive and time-consuming,
intermediate results are saved to disk and loaded as needed.
"""

import os
import copy
import time
import itertools
from pathlib import Path
import multiprocessing
from functools import partial
import statistics as stat

import numpy as np
from numpy import ma
from scipy import stats
from sklearn.model_selection import StratifiedKFold, GroupKFold
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from psiz.models import Inverse, Exponential, HeavyTailed, StudentsT
from psiz.models import load_embedding
from psiz.dimensionality import dimension_search
from psiz.simulate import Agent
from psiz.generator import RandomGenerator, ActiveGenerator
from psiz import trials
from psiz import datasets
from psiz import visualize
from psiz.utils import similarity_matrix, matrix_comparison, assess_convergence


def main(fp_results):
    """Script to run all experiments.

    Preliminary A:  Ground truth embedding.
        An initial embedding is inferred using real observations data.
        This embedding is then treated as ground truth for the purpose
        of simulating human behavior in Experiment 2.

    Preliminary B:  Assess convergence of embedding.

    Experiment 1: Similarity kernel comparison using real human data.

    Simulation 1: Trial configuration comparison.
        Two different trial configurations are compared using
            simulations. In addition, random and active selection are
            compared using simulations.

    Simulation 2:  Independent versus shared embedding inference for
        two groups.

    Notes:
        In Simulation 1 and 2, the dimensionality is not inferred at
            each step. Instead, the correct dimensionality is assumed
            to be known.
        In Simulation 1, result files are saved after each run. New
            results are appeneded to existing results if the respective
            file already exists.

    """
    # Settings.
    domain = 'birds'

    # Define experiment/simulation filepaths.
    fp_pre_domain = fp_results / Path('prelim/{0:s}'.format(domain))
    fp_exp_1_domain = fp_results / Path('exp_1/{0:s}'.format(domain))
    fp_sim_1_domain = fp_results / Path('sim_1/{0:s}'.format(domain))
    fp_sim_2 = fp_results / Path('sim_2')

    # Run preliminaries.
    run_prelim_a(domain, fp_pre_domain)
    run_prelim_b(domain, fp_pre_domain)

    # Run each experiment/simulation.
    run_exp_1(domain, fp_pre_domain, fp_exp_1_domain)
    run_sim_1(domain, fp_pre_domain, fp_sim_1_domain)
    run_sim_2(domain, fp_sim_2)

    # Visualize Experiment 1.
    fp_fig_exp_1 = fp_exp_1_domain / Path('exp_1.pdf')
    visualize_exp_1(fp_exp_1_domain, fp_fig_exp_1)

    # Visualize Simulation 1.
    fp_fig_sim_1 = fp_sim_1_domain / Path('sim_1.pdf')
    fp_data_r2c1 = fp_sim_1_domain / Path('r2c1/r2c1_data.p')
    fp_data_r8c2 = fp_sim_1_domain / Path('r8c2/r8c2_data.p')
    fp_data_a8c2 = fp_sim_1_domain / Path('a8c2/a8c2_data.p')
    data_r2c1 = pickle.load(open(fp_data_r2c1, 'rb'))
    data_r8c2 = pickle.load(open(fp_data_r8c2, 'rb'))
    data_a8c2 = pickle.load(open(fp_data_a8c2, 'rb'))
    visualize_exp_2(data_r2c1, data_r8c2, data_a8c2, fp_fig_sim_1)

    # Visualize Simulation 2.
    fp_fig_sim_2 = fp_sim_2 / Path('sim_2.pdf')
    fp_data_r2c1 = fp_sim_2 / Path('r8c2_1g/r8c2_1g_data.p')
    fp_data_r8c2 = fp_sim_2 / Path('r8c2_2g/r8c2_2g_data.p')
    data_r8c2_g1 = pickle.load(open(fp_data_r2c1, 'rb'))
    data_r8c2_g2 = pickle.load(open(fp_data_r8c2, 'rb'))
    visualize_exp_3(data_r8c2_g1, data_r8c2_g2, fp_fig_sim_2)


def run_prelim_a(domain, fp_pre_domain):
    """Run Experiment 0."""
    # Settings.
    freeze_options = {'theta': {'rho': 2., 'tau': 1.}}
    # Define filepaths.
    fp_emb_true_2d = fp_pre_domain / Path('emb_true_2d.hdf5')
    fp_fig_emb_true_2d = fp_pre_domain / Path('emb_true_2d.pdf')
    fp_emb_true = fp_pre_domain / Path('emb_true.hdf5')

    # Load the real observations.
    (obs, catalog) = datasets.load_dataset('birds-16')

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
        fname=fp_fig_emb_true_2d
    )

    np.random.seed(123)
    # Determine solution by selecting dimensionality based on cross-validation.
    summary = dimension_search(
        obs, Exponential, catalog.n_stimuli, freeze_options=freeze_options,
        n_restart=20, n_split=3, n_fold=3,
        max_patience=1, verbose=1
    )
    n_dim = summary["dim_best"]

    emb_true = Exponential(catalog.n_stimuli, n_dim)
    emb_true.freeze(freeze_options)
    emb_true.fit(obs, n_restart=100)
    emb_true.save(fp_emb_true)


def run_prelim_b(domain, fp_pre_domain):
    """Run Experiment 0b.

    Assess convergence of embedding.
    """
    # Settings.
    fp_probe = fp_pre_domain / Path('probe.p')
    fp_fig_probe = fp_pre_domain / Path('probe.pdf')

    # Load the real observations.
    (obs, catalog) = datasets.load_dataset('birds-16')

    converge_data = assess_convergence(
        obs, Exponential, catalog.n_stimuli, 3, n_partition=10, n_back=3,
        n_restart=50, score='pearson', verbose=2
    )
    pickle.dump(converge_data, open(fp_probe, "wb"))
    # converge_data = pickle.load(open(fp_probe, 'rb'))
    visualize.visualize_convergence(converge_data, fp_fig_probe)


def run_exp_1(domain, fp_pre_domain, fp_exp_1_domain):
    """Run Experiment 1."""
    # Settings.
    n_fold = 10

    # Load the real observations.
    (obs, catalog) = datasets.load_dataset('birds-16')

    # Instantiate the balanced k-fold cross-validation object.
    np.random.seed(seed=4352)
    skf = GroupKFold(n_splits=n_fold)
    split_list = list(
        skf.split(obs.stimulus_set, obs.config_idx, obs.agent_id)
    )
    n_fold = skf.get_n_splits()

    # Inverse
    fp_model = fp_exp_1_domain / Path('Inverse')
    freeze_options = {
        'theta': {
            'rho': 2,
            'tau': 1
        }
    }
    loss = embedding_cv(
        split_list, obs, Inverse, catalog.n_stimuli, freeze_options)
    pickle.dump(loss, open(str(fp_model / Path("loss.p")), "wb"))

    # Exponential family.
    fp_model = fp_exp_1_domain / Path('Exponential')
    freeze_options = {}
    loss = embedding_cv(
        split_list, obs, Exponential, catalog.n_stimuli, freeze_options)
    pickle.dump(loss, open(str(fp_model / Path("loss.p")), "wb"))

    # Heavy-tailed family.
    fp_model = fp_exp_1_domain / Path('HeavyTailed')
    freeze_options = {}
    loss = embedding_cv(
        split_list, obs, HeavyTailed, catalog.n_stimuli, freeze_options)
    pickle.dump(loss, open(str(fp_model / Path("loss.p")), "wb"))

    # Student-t family.
    fp_model = fp_exp_1_domain / Path('StudentsT')
    freeze_options = {}
    loss = embedding_cv(
        split_list, obs, StudentsT, catalog.n_stimuli, freeze_options)
    pickle.dump(loss, open(str(fp_model / Path("loss.p")), "wb"))


def dim_from_list(dim_list):
    """Estimate dimensionality from list."""
    try:
        n_dim = stat.mode(dim_list),
        n_dim = n_dim[0]
    except stat.StatisticsError:
        n_dim = np.ceil(np.mean(dim_list))
    return int(n_dim)


def run_sim_1(domain, fp_pre_domain, fp_sim_1_domain):
    """Run Experiment 2."""
    # Settings.
    freeze_options = {'theta': {'rho': 2., 'tau': 1.}}
    seed_list = [913, 192, 785, 891, 841]
    time_s_2c1 = 3.06
    time_s_8c2 = 8.98

    # Define experiment conditions.
    cond_info_r2c1 = {
        'name': 'Random 2-choose-1',
        'prefix': 'r2c1',
        'domain': domain,
        'query_policy': 'kl',
        'selection_policy': 'random',
        'n_reference': 2,
        'n_select': 1,
        'n_trial_initial': 500,
        'n_trial_total': 150500,
        'n_trial_per_round': 5000,
        'time_s_per_trial': time_s_2c1,
        'n_redundant': 1
    }
    cond_info_r8c2 = {
        'name': 'Random 8-choose-2',
        'prefix': 'r8c2',
        'domain': domain,
        'query_policy': 'kl',
        'selection_policy': 'random',
        'n_reference': 8,
        'n_select': 2,
        'n_trial_initial': 50,
        'n_trial_total': 15050,
        'n_trial_per_round': 500,
        'time_s_per_trial': time_s_8c2,
        'n_redundant': 1
    }
    cond_info_a8c2 = {
        'name': 'Active 8-choose-2',
        'prefix': 'a8c2',
        'domain': domain,
        'query_policy': 'kl',
        'selection_policy': 'active',
        'n_reference': 8,
        'n_select': 2,
        'n_trial_initial': 50,
        'n_trial_total': 10050,
        'n_trial_per_round': 40,
        'time_s_per_trial': time_s_8c2,
        'n_redundant': 1
    }

    fp_emb_true = fp_pre_domain / Path('emb_true.hdf5')
    emb_true = load_embedding(fp_emb_true)

    simulate_multiple_runs(
        seed_list, emb_true, cond_info_r2c1, freeze_options,
        fp_sim_1_domain / Path(cond_info_r2c1['prefix'])
    )

    simulate_multiple_runs(
        seed_list, emb_true, cond_info_r8c2, freeze_options,
        fp_sim_1_domain / Path(cond_info_r8c2['prefix'])
    )

    simulate_multiple_runs(
        seed_list, emb_true, cond_info_a8c2, freeze_options,
        fp_sim_1_domain / Path(cond_info_a8c2['prefix'])
    )


def run_sim_2(domain, fp_sim_2):
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

    simulate_multiple_runs(
        seed_list, emb_true, cond_info_r8c2_1g, freeze_options,
        fp_sim_2 / Path(cond_info_r8c2_1g['prefix']), group_id=1
    )

    simulate_multiple_runs(
        seed_list, emb_true, cond_info_r8c2_2g, freeze_options,
        fp_sim_2 / Path(cond_info_r8c2_2g['prefix']), group_id=1,
        obs_existing=obs_novice
    )


def embedding_cv(
        split_list, obs, embedding_constructor, n_stimuli, freeze_options):
    """Embedding cross-validation procedure."""
    # Cross-validation settings.
    verbose = 2
    n_fold = len(split_list)
    loss_train = np.empty((n_fold))
    loss_test = np.empty((n_fold))

    loaded_fold = partial(
        evaluate_fold, split_list=split_list, obs=obs,
        embedding_constructor=embedding_constructor, n_stimuli=n_stimuli,
        freeze_options=freeze_options, verbose=verbose)

    fold_list = range(n_fold)
    # for i_fold in fold_list:
    #     (loss_train[i_fold], loss_test[i_fold]) = loaded_func(i_fold)
    with multiprocessing.Pool() as pool:
        results = pool.map(loaded_fold, fold_list)

    loss_train = []
    loss_test = []
    acc_2c1 = []
    acc_8c2 = []
    n_dim = []
    for i_fold in fold_list:
        loss_train.append(results[i_fold][0])
        loss_test.append(results[i_fold][1])
        acc_2c1.append(results[i_fold][2])
        acc_8c2.append(results[i_fold][3])
        n_dim.append(results[i_fold][4])

    res = {
        'train': loss_train, 'test': loss_test, 'n_dim': n_dim,
        '2c1': acc_2c1, '8c2': acc_8c2
    }
    return res


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
    summary = dimension_search(
        obs_train, embedding_constructor, n_stimuli, n_restart=n_restart_dim,
        freeze_options=freeze_options, n_split=3, n_fold=3, max_patience=1
    )
    n_dim = summary["dim_best"]
    if verbose > 1:
        print("        Suggested dimensionality: {0}".format(n_dim))

    # Instantiate model.
    emb = embedding_constructor(n_stimuli, n_dim, n_group)
    if len(freeze_options) > 0:
        emb.freeze(freeze_options)
    # Fit model using training data.
    loss_train, _ = emb.fit(obs_train, n_restart=n_restart_fit)

    # Test.
    obs_test = obs.subset(test_index)
    loss_test = emb.evaluate(obs_test)

    # Top N analysis.
    locs_2c1 = np.equal(obs_test.n_select, 1)
    obs_test_2c1 = obs_test.subset(locs_2c1)

    locs_8c2 = np.equal(obs_test.n_select, 2)
    obs_test_8c2 = obs_test.subset(locs_8c2)

    prob_2c1 = emb.outcome_probability(obs_test_2c1)
    prob_8c2 = emb.outcome_probability(obs_test_8c2)

    acc_2c1 = top_n_accuracy(prob_2c1, 1)
    acc_8c2 = top_n_accuracy(prob_8c2, 5)

    return (loss_train, loss_test, acc_2c1, acc_8c2, n_dim, emb)


def top_n_accuracy(prob, n_top):
    """Return the top-N accuracy."""
    n_trial = prob.shape[0]

    acc = np.zeros(n_trial)
    for i_trial in range(n_trial):
        sorted_idx = np.argsort(-prob[i_trial, :])
        top_idx = sorted_idx[0:n_top]
        if np.sum(np.equal(top_idx, 0)) > 0:
            acc[i_trial] = 1
    acc = np.mean(acc)
    return acc


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
    fp_data_run = dir_cond / Path('{0:s}_{1:s}_data.p'.format(
        cond_info['prefix'], run_id)
    )
    fp_obs = dir_cond / Path('{0:s}_{1:s}_obs.hdf5'.format(
        cond_info['prefix'], run_id)
    )
    # fp_emb_inf = dir_cond / Path('{0:s}_{1:s}_emb_inf.hdf5'.format(
    #     cond_info['prefix'], run_id)
    # )

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
        loss[i_round], _ = emb_inferred.fit(
            obs.subset(include_idx), n_restart=50, init_mode=init_mode)
        # Compare the inferred model with ground truth by comparing the
        # similarity matrices implied by each model.
        simmat_infer = similarity_matrix(
            emb_inferred.similarity, emb_inferred.z['value'])
        r_squared[i_round] = matrix_comparison(simmat_infer, simmat_true)
        is_valid[i_round] = True
        print(
            'Round {0} ({1:d} trials) | Loss: {2:.2f} | r^2:{3:.3f}'.format(
                i_round, int(n_trial[i_round]), loss[i_round],
                r_squared[i_round]
            )
        )
        results_temp = {
            'n_trial': np.expand_dims(n_trial[0:i_round + 1], axis=1),
            'loss': np.expand_dims(loss[0:i_round + 1], axis=1),
            'r_squared': np.expand_dims(r_squared[0:i_round + 1], axis=1),
            'is_valid': np.expand_dims(is_valid[0:i_round + 1], axis=1)
        }
        data = {'info': cond_info, 'results': results_temp}
        pickle.dump(data, open(fp_data_run, 'wb'))
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
        emb_true, cond_info, freeze_options, dir_cond, run_id, group_id=0,
        n_redundant=1):
    """Simulate active selection progress for a trial configuration.

    Record:
        n_trial, loss, r^2
    """
    # Define filepaths.
    fp_data_run = dir_cond / Path('{0:s}_{1:s}_data.p'.format(
        cond_info['prefix'], run_id)
    )
    fp_obs = dir_cond / Path('{0:s}_{1:s}_obs.hdf5'.format(
        cond_info['prefix'], run_id)
    )
    fp_emb_inf = dir_cond / Path('{0:s}_{1:s}_emb_inf.hdf5'.format(
        cond_info['prefix'], run_id)
    )

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

    # The first round is seeded using a randomly generated docket.
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
    loss[i_round], _ = emb_inferred.fit(obs, n_restart=50, init_mode='cold')
    simmat_infer = similarity_matrix(
        emb_inferred.similarity, emb_inferred.z['value'])
    r_squared[i_round] = matrix_comparison(simmat_infer, simmat_true)
    is_valid[i_round] = True
    print(
        'Round {0} ({1:d} trials) | Loss: {2:.2f} | r^2:{3:.3f}'.format(
            i_round, int(n_trial[i_round]), loss[i_round],
            r_squared[i_round]
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
            obs, n_final_sample=1000, n_burn=100, thin_step=10
        )
        elapsed = time.time() - time_start
        print('Posterior | Elapsed time: {0:.2f} m'.format(elapsed / 60))
        emb_inferred.z['value'] = np.median(samples['z'], axis=2)

        time_start = time.time()
        active_docket, _ = active_gen.generate(
            cond_info['n_trial_per_round'], emb_inferred, samples
        )
        elapsed = time.time() - time_start
        print('Active | Elapsed time: {0:.2f} m'.format(elapsed / 60))
        # Simulate observations.
        for _ in range(n_redundant):
            new_obs = agent.simulate(active_docket)
            obs = trials.stack([obs, new_obs])
        n_trial[i_round] = obs.n_trial

        if np.mod(i_round, 5) == 0:
            # Infer new embedding with exact restarts.
            freeze_options = {'z': emb_inferred.z['value']}
            emb_inferred.freeze(freeze_options)
            loss[i_round], _ = emb_inferred.fit(
                obs, n_restart=3, init_mode='exact')
            emb_inferred.thaw(['z'])
            loss[i_round], _ = emb_inferred.fit(
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
            'Round {0} ({1:d} trials) | Loss: {2:.2f} | r^2:{3:.3f}'.format(
                i_round, int(n_trial[i_round]), loss[i_round],
                r_squared[i_round]
            )
        )
        results_temp = {
            'n_trial': np.expand_dims(n_trial[0:i_round + 1], axis=1),
            'loss': np.expand_dims(loss[0:i_round + 1], axis=1),
            'r_squared': np.expand_dims(r_squared[0:i_round + 1], axis=1),
            'is_valid': np.expand_dims(is_valid[0:i_round + 1], axis=1)
        }
        data = {'info': cond_info, 'results': results_temp}
        pickle.dump(data, open(fp_data_run, 'wb'))
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
            'w': attention
        }
    }
    emb.freeze(freeze_options)
    return emb


def simulate_run_random_existing(
        emb_true, cond_info, freeze_options, dir_cond, run_id,
        group_id, obs_existing):
    """Simulate random selection progress for a trial configuration.

    Record:
        n_trial, loss, r^2
    """
    # Define filepaths.
    fp_data_run = dir_cond / Path('{0:s}_{1:s}_data.p'.format(
        cond_info['prefix'], run_id)
    )
    fp_obs = dir_cond / Path('{0:s}_{1:s}_obs.hdf5'.format(
        cond_info['prefix'], run_id)
    )
    # fp_emb_inf = dir_cond / Path('{0:s}_{1:s}_emb_inf.hdf5'.format(
    #     cond_info['prefix'], run_id)
    # )

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

        loss[i_round], _ = emb_inferred.fit(
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
            'Round {0} ({1:d} trials) | Loss: {2:.2f} | r^2:{3:.3f}'.format(
                i_round, int(n_trial[i_round]), loss[i_round],
                r_squared[i_round]
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
        pickle.dump(data, open(fp_data_run, 'wb'))

    results = {
        'n_trial': np.expand_dims(n_trial, axis=1),
        'loss': np.expand_dims(loss, axis=1),
        'r_squared': np.expand_dims(r_squared, axis=1),
        'is_valid': np.expand_dims(is_valid, axis=1)
    }
    return results


def visualize_exp_1(fp_cv, fp_figure=None):
    """Plot results."""
    # Settings.
    model_list = ['Inverse', 'Exponential', 'HeavyTailed', 'StudentsT']
    pretty_list = ['Inverse', 'Exponential', 'Heavy-Tailed', 'Student-t']
    # Source
    rgb1 = np.array((0.0, 0.0, 0.5312, 1.0))
    rgb2 = np.array((1.0, 0.8125, 0.0, 1.0))
    rgb3 = np.array((0.5, 0.0, 0.0, 1.0))
    # Lighter version.
    color_scale = .4  # Lower scale yeilds lighter colors.
    rgb1_light = 1 - (color_scale * (1 - rgb1))
    rgb2_light = 1 - (color_scale * (1 - rgb2))
    rgb3_light = 1 - (color_scale * (1 - rgb3))
    fontdict = {
        'fontsize': 12,
        'verticalalignment': 'center',
        'horizontalalignment': 'center',
        'fontweight': 'bold'
    }

    n_model = len(model_list)
    train_mu = np.empty(n_model)
    train_se = np.empty(n_model)
    test_mu = np.empty(n_model)
    test_sem = np.empty(n_model)
    test_sd = np.empty(n_model)

    acc_2c1_mu = np.empty(n_model)
    acc_2c1_sem = np.empty(n_model)
    acc_8c2_mu = np.empty(n_model)
    acc_8c2_sem = np.empty(n_model)

    loss_all = []
    acc_2c1_all = []
    acc_8c2_all = []
    for i_model, model_name in enumerate(model_list):
        filepath = fp_cv / Path(model_name, 'loss.p')
        loss = pickle.load(open(str(filepath), "rb"))
        loss_all.append(loss['test'])
        acc_2c1_all.append(loss['2c1'])
        acc_8c2_all.append(loss['8c2'])

        train_mu[i_model] = np.mean(loss['train'])
        train_se[i_model] = stats.sem(loss['train'])

        test_mu[i_model] = np.mean(loss['test'])
        test_sem[i_model] = stats.sem(loss['test'])
        test_sd[i_model] = np.std(loss['test'])
        ndim_mode = dim_from_list(loss["n_dim"])

        acc_2c1_mu[i_model] = np.mean(loss['2c1'])
        acc_2c1_sem[i_model] = stats.sem(loss['2c1'])

        acc_8c2_mu[i_model] = np.mean(loss['8c2'])
        acc_8c2_sem[i_model] = stats.sem(loss['8c2'])

        print(
            '{0:s} n_dim | mode={1}, mean={2}'.format(
                model_name, ndim_mode, np.mean(loss["n_dim"])
            )
        )
    print('\n')

    complete_pairwise_ttest(model_list, loss_all)
    complete_pairwise_ttest(model_list, acc_2c1_all)
    complete_pairwise_ttest(model_list, acc_8c2_all)

    # Determine the maximum and minimum y values in the figure.
    ymin = np.min(np.stack((train_mu - train_se, test_mu - test_sem), axis=0))
    ymax = np.max(np.stack((train_mu + train_se, test_mu + test_sem), axis=0))
    ydiff = ymax - ymin
    ypad = ydiff * .1
    limits_loss = [ymin - ypad, ymax + ypad]

    _, ax = plt.subplots(figsize=(6.5, 3))
    xg = np.arange(n_model)

    # Loss.
    width = 0.75
    ax = plt.subplot(1, 2, 1)
    color = rgb1
    error_kw = {'linewidth': 4, 'ecolor': rgb1_light}
    _ = ax.bar(
        xg, test_mu, yerr=test_sem,
        width=width, color='b', error_kw=error_kw
    )
    ax.set_ylabel('Loss')
    ax.set_title('Validation Loss')
    ax.set_xticks(xg)
    ax.set_xticklabels(pretty_list, rotation=45)
    ax.set_ylim(limits_loss)
    ax.text(
        -.2, 1.07, "(A)", fontdict=fontdict,
        horizontalalignment='center', verticalalignment='center',
        transform=ax.transAxes
    )

    # Top-N Accuracy
    width = 0.35
    ax = plt.subplot(1, 2, 2)
    color = rgb2
    error_kw = {'linewidth': 6, 'ecolor': rgb2_light}
    _ = ax.bar(
        xg - width/2, acc_2c1_mu, yerr=acc_2c1_sem,
        width=width, color=color, error_kw=error_kw, label='Top-1 2-choose-1'
    )
    color = rgb3
    error_kw = {'linewidth': 6, 'ecolor': rgb3_light}
    _ = ax.bar(
        xg + width/2, acc_8c2_mu, yerr=acc_8c2_sem,
        width=width, color=color, error_kw=error_kw, label='Top-5 8-choose-2'
    )
    ax.set_ylabel('Top-N Accuracy')
    ax.set_title('Top-N Validation Accuracy')
    ax.set_xticks(xg)
    ax.set_xticklabels(pretty_list, rotation=45)
    ax.legend(loc=3, framealpha=0.95)
    ax.set_ylim([.5, .8])
    ax.text(
        -0.2, 1.07, "(B)", fontdict=fontdict,
        horizontalalignment='center', verticalalignment='center',
        transform=ax.transAxes
    )

    plt.tight_layout()
    if fp_figure is None:
        plt.show()
    else:
        plt.savefig(
            os.fspath(fp_figure), format='pdf', bbox_inches="tight", dpi=100
        )


def complete_pairwise_ttest(model_list, val, alpha=.05):
    """Perform all possible pairwise comparisons."""
    n_model = len(model_list)

    for i_model in range(n_model):
        print(
            '{0:s} | M={1:.2f}, SD={2:.2f}'.format(
                model_list[i_model],
                np.mean(val[i_model]),
                np.std(val[i_model])
            )
        )

    # Pairwise test.
    comb_list = list(itertools.combinations(range(n_model), 2))
    comb_list = np.array(comb_list)
    n_comb = len(comb_list)

    # Adjusted alpha value.
    alpha_adj = .05 / n_comb
    print("Bonferroni corrected alpha: {0:.4}".format(alpha_adj))

    for i_comb in range(n_comb):
        idx_a = comb_list[i_comb, 0]
        idx_b = comb_list[i_comb, 1]
        print_ttest(
            model_list[idx_a], model_list[idx_b],
            val[idx_a], val[idx_b],
            alpha_adj
        )
    print('\n')


def print_ttest(m1, m2, x1, x2, alpha, verbose=0):
    """Print t-test results in APA format."""
    [t, p] = stats.ttest_ind(x1, x2)
    is_significant = p < alpha
    if is_significant:
        msg = "SIGNIFICANT"
    else:
        msg = ""

    if (verbose > 0) or is_significant:
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
            '{0}:{1} | t({2}) = {3:.2f}, p = {4:.2f} {5}'.format(
                m1, m2, df, t, p, msg
            )
        )
        print('\n')


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

    _, ax = plt.subplots(1, 1, figsize=(6, 4))

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
            os.fspath(fp_figure), format='pdf', bbox_inches="tight", dpi=100
        )


def plot_exp2_condition(
        ax, data, c_line, c_env, c_scatter, fontdict, rsquared_crit=.95,
        thin_step=1):
    """Plot condition."""
    legend_name = data['info']['name']
    results = data['results']

    if 'is_valid' in results:
        results["is_valid"] = results["is_valid"][0::thin_step, :]
    results["loss"] = results["loss"][0::thin_step, :]
    results["n_trial"] = results["n_trial"][0::thin_step, :]
    results["r_squared"] = results["r_squared"][0::thin_step, :]

    time_factor = data['info']['time_s_per_trial'] / (60 * 60)

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
        xg, r_mean_avg, '-', linewidth=1, color=c_line,
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
            np.log10(time_factor * trial_thresh), r2_thresh, marker='d',
            color=c_scatter, edgecolors='k'
        )
        ax.text(
            np.log10(time_factor * trial_thresh), r2_thresh + .04,
            "{0:.1f}".format(time_factor * trial_thresh), fontdict=fontdict
        )
    # Major ticks.
    ax.set_xticks([-1, 0, 1, 2])
    ax.set_xticklabels([".1", "1", "10", "100"])
    # Minor ticks.
    lg = np.hstack(
        (np.arange(.1, 1, .1), np.arange(1, 10, 1), np.arange(10, 100, 10))
    )
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
    rgb3_light = 1 - (color_scale * (1 - rgb3))

    fontdict_bold = {
        'fontsize': 12,
        'verticalalignment': 'center',
        'horizontalalignment': 'center',
        'fontweight': 'bold'
    }
    fontdict_white = {
        'fontsize': 10,
        'verticalalignment': 'center',
        'horizontalalignment': 'center',
        'color': "white"
    }

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

    _, ax = plt.subplots(1, 2, figsize=(6, 4))

    ax = plt.subplot(1, 2, 2)
    i_cond = 0
    (time_thresh_ind, _) = plot_exp3_condition(
        ax, data_r8c2_g1, c_line[i_cond], c_env[i_cond], c_scatter[i_cond],
        fontdict, 'Independent'
    )

    i_cond = 1
    (time_thresh_shared, _) = plot_exp3_condition(
        ax, data_r8c2_g2, c_line[i_cond], c_env[i_cond], c_scatter[i_cond],
        fontdict, 'Shared'
    )

    ax.set_ylim(bottom=0., top=1.05)
    ax.set_xlabel('Total Expert Worker Hours')
    ax.set_xticks([.1, 1, 10])
    ax.set_xticklabels([.1, 1, 10])
    ax.set_ylabel(r'Pearson $\rho$')
    ax.legend()
    ax.set_title("Expert Hours")
    ax.text(
        -.2, 1.07, "(B)", fontdict=fontdict_bold,
        horizontalalignment='center', verticalalignment='center',
        transform=ax.transAxes
    )

    # Total worker hours.
    ax = plt.subplot(1, 2, 1)
    yg_nov = np.array([time_thresh_ind, (3550 * 8.98) / (60 * 60)])
    yg_exp = np.array([time_thresh_ind, time_thresh_shared])

    ax.bar(0, yg_nov[0], color=rgb1_light, label="Independent Novice")
    ax.bar(1, yg_nov[1], color=rgb3_light, label="Shared Novice")
    ax.bar(
        0, yg_exp[0], bottom=yg_nov[0], color=rgb1, label="Independent Expert"
    )
    ax.bar(
        1, yg_exp[1], bottom=yg_nov[1], color=rgb3, label="Shared Expert"
    )

    ax.set_ylabel('Total Worker Hours')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Independent", "Shared"])
    ax.set_title("Total Hours")
    ax.text(
        0, yg_nov[0] + .5 * yg_exp[0], "{0:.1f}".format(yg_exp[0]),
        horizontalalignment='center', verticalalignment='center',
        fontdict=fontdict_white
    )
    ax.text(
        1, yg_nov[1] + .5 * yg_exp[1], "{0:.1f}".format(yg_exp[1]),
        horizontalalignment='center', verticalalignment='center',
        fontdict=fontdict_white
    )
    ax.text(
        -.2, 1.07, "(A)", fontdict=fontdict_bold,
        horizontalalignment='center', verticalalignment='center',
        transform=ax.transAxes
    )
    ax.legend(loc=3, framealpha=0.95)

    plt.tight_layout()

    if fp_figure is None:
        plt.show()
    else:
        plt.savefig(
            os.fspath(fp_figure), format='pdf', bbox_inches="tight", dpi=100
        )


def plot_exp3_condition(
        ax, data, c_line, c_env, c_scatter, fontdict, legend_name,
        rsquared_crit=.95):
    """Plot condition."""
    results = data['results']

    time_s_8c2 = 8.98
    time_factor = time_s_8c2 / (60 * 60)

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
    time_thresh = None
    r2_thresh = None
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
        time_thresh = time_factor * trial_thresh
        ax.scatter(
            time_thresh, r2_thresh, marker='d', color=c_scatter,
            edgecolors='k')
        ax.text(
            time_thresh, r2_thresh + .06, "{0:.1f}".format(
                time_factor * trial_thresh
            ),
            fontdict=fontdict)
        return (time_thresh, r2_thresh)


if __name__ == "__main__":
    # Specify the path to a folder where you would like to store all your
    # results by change the following line. For example,
    # fp_results = Path('/home/results').
    fp_project = Path.home() / Path('projects/psiz-projects/psiz-brm/results')
    main(fp_results)
