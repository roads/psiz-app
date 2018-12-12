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
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from psiz.models import Exponential, HeavyTailed, StudentsT, load_embedding
from psiz.dimensionality import suggest_dimensionality
from psiz.simulate import Agent
from psiz.generator import RandomGenerator, ActiveGenerator
from psiz import trials
from psiz import datasets
from psiz import visualize
from psiz.utils import similarity_matrix, matrix_comparison

# from psiz.trials import Observations, load_trials
# import psiz.utils as ut

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
        fp_exp3_domain = fp_results / Path('exp_3/{0:s}'.format(domain))
        # Run each experiment.
        # run_exp_0(domain, fp_exp0_domain)
        # run_exp_1(domain, fp_exp0_domain, fp_exp1_domain)
        run_exp_2(domain, fp_exp0_domain, fp_exp2_domain)
        # run_exp_3(domain, fp_exp0_domain, fp_exp3_domain)

    # Visualize Experiment 1 Results.
    # fp_cv = fp_results / Path('exp_1/{0:s}'.format(domain))
    # fp_figure_exp1 = fp_results / Path('exp_1/{0:s}/exp1.pdf'.format(domain))
    # visualize_exp_1(fp_cv, fp_figure_exp1)

    # Visualize Experiment 2 Results.
    # Define filepaths.
    # fp_data_r2c1 = fp_results / Path('exp_2/{0:s}/r2c1/r2c1_data.p'.format(domain))
    # fp_data_r8c2 = fp_results / Path('exp_2/{0:s}/r8c2/r8c2_data.p'.format(domain))
    # fp_data_a8c2 = fp_results / Path('exp_2/{0:s}/a8c2/a8c2_data.p'.format(domain))
    # fp_figure_exp2 = fp_results / Path('exp_2/{0:s}/exp2.pdf'.format(domain))
    # # Load data.
    # data_r2c1 = pickle.load(open(fp_data_r2c1, 'rb'))
    # data_r8c2 = pickle.load(open(fp_data_r8c2, 'rb'))
    # data_a8c2 = pickle.load(open(fp_data_a8c2, 'rb'))
    # visualize_exp_2(data_r2c1, data_r8c2, data_a8c2, fp_figure_exp2)

    # Visualize Experiment 3 Results.


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


def run_exp_1(domain, fp_exp0_domain, fp_exp1_domain):
    """Run Experiment 0."""
    # Settings.
    n_fold = 10

    # Load the real observations.
    if domain == 'birds':
        dataset_name = 'birds-16'
    elif domain == 'rocks':
        dataset_name = 'rocks_Nosofsky_etal_2016'
    (obs, catalog) = datasets.load_dataset(dataset_name)

    # Instantiate the balanced k-fold cross-validation object.
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=723)

    # Exponential family.
    fp_model = fp_exp1_domain / Path('Exponential')
    freeze_options = {}
    # freeze_options = dict(rho=2., beta=10.)
    loss = embedding_cv(
        skf, obs, Exponential, catalog.n_stimuli, freeze_options)
    pickle.dump(loss, open(str(fp_model / Path("loss.p")), "wb"))

    # Gaussian family.
    # fp_model = fp_exp1_domain / Path('Gaussian')
    # freeze_options = dict(rho=2., tau=2., gamma=0.)
    # loss = embedding_cv(
    #     skf, obs, Exponential, catalog.n_stimuli, freeze_options)
    # pickle.dump(loss, open(str(fp_model / Path("loss.p")), "wb"))

    # Laplacian family.
    # fp_model = fp_exp1_domain / Path('Laplacian')
    # freeze_options = dict(rho=2., tau=1., gamma=0.)
    # loss = embedding_cv(
    #     skf, obs, Exponential, catalog.n_stimuli, freeze_options)
    # pickle.dump(loss, open(str(fp_model / Path("loss.p")), "wb"))

    # Heavy-tailed family.
    fp_model = fp_exp1_domain / Path('HeavyTailed')
    freeze_options = {}
    loss = embedding_cv(
        skf, obs, HeavyTailed, catalog.n_stimuli, freeze_options)
    pickle.dump(loss, open(str(fp_model / Path("loss.p")), "wb"))

    # Student-t family.
    fp_model = fp_exp1_domain / Path('StudentsT')
    freeze_options = {}
    loss = embedding_cv(
        skf, obs, StudentsT, catalog.n_stimuli, freeze_options)
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
        'n_trial_initial': 50,  #500,
        'n_trial_total': 15050,  #15000,
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

    fp_emb_true = fp_exp0_domain / Path('emb_true.hdf5')
    emb_true = load_embedding(fp_emb_true)

    # simulate_multiple_runs(
    #     seed_list, emb_true, cond_info_r2c1, freeze_options, fp_exp2_domain / Path(cond_info_r2c1['prefix'])
    # )

    # simulate_multiple_runs(
    #     seed_list, emb_true, cond_info_r8c2, freeze_options, fp_exp2_domain / Path(cond_info_r8c2['prefix'])
    # )

    simulate_multiple_runs(
        seed_list, emb_true, cond_info_a8c2, freeze_options, fp_exp2_domain / Path(cond_info_a8c2['prefix'])
    )


def run_exp_3(domain, fp_exp0_domain, fp_exp3_domain):
    """Run Experiment 2.

    Random 8-choose-2 novices
    Random 8-choose-2 experts
    Active 8-choose-2 experts

    Generate all random novice data
    Generate all random expert data

    Infer joint embedding with all random novice + incremental random expert
    Infer joint embedding with all random novice + incremental active expert
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
        'n_trial_initial': 6010,
        'n_trial_total': 6010,
        'n_trial_per_round': 250,
    }

    cond_info_r8c2_2g = {
        'name': 'Random 8-choose-2',
        'prefix': 'r8c2_2g',
        'domain': domain,
        'selection_policy': 'random',
        'n_reference': 8,
        'n_select': 2,
        'n_trial_initial': 10,
        'n_trial_total': 6010,
        'n_trial_per_round': 250,
    }

    np.random.seed(548)
    emb_true = exp_3_ground_truth(n_stimuli, n_dim, n_group)
    # fp_emb_true = fp_exp0_domain / Path('emb_true.hdf5')
    # emb_true = load_embedding(fp_emb_true)

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

    # TODO for loop
    run_id = str(seed_list[0])

    results = simulate_run_random(
        emb_true, cond_info_r8c2_1g, freeze_options, fp_exp3_domain, run_id,
        group_id=1
    )

    # results = simulate_run_random_exp3(
    #     emb_true, cond_info_r8c2_2g, freeze_options, fp_exp3_domain, run_id,
    #     obs_novice
    # )


def embedding_cv(skf, obs, embedding_constructor, n_stimuli, freeze_options):
    """Embedding cross-validation procedure."""
    # Cross-validation settings.
    verbose = 2
    n_fold = skf.get_n_splits()

    J_train = np.empty((n_fold))
    J_test = np.empty((n_fold))

    split_list = list(skf.split(obs.stimulus_set, obs.config_idx))
    # loaded_fold = lambda i_fold: evaluate_fold(i_fold, split_list, displays, display_info, embedding_constructor, n_stimuli, freeze_options, verbose)
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
    for i_fold in fold_list:
        J_train.append(results[i_fold][0])
        J_test.append(results[i_fold][1])

    return {'train': J_train, 'test': J_test}


def evaluate_fold(
        i_fold, split_list, obs, embedding_constructor, n_stimuli,
        freeze_options, verbose):
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
        obs_train, embedding_constructor, n_stimuli, n_restart=n_restart_dim)
    # Instantiate model.
    embedding_model = embedding_constructor(n_stimuli, n_dim, n_group)
    if len(freeze_options) > 0:
        embedding_model.freeze(**freeze_options)
    # Fit model using training data.
    J_train = embedding_model.fit(obs_train, n_restart=n_restart_fit)

    # Test.
    obs_test = obs.subset(test_index)
    J_test = embedding_model.evaluate(obs_test)

    return (J_train, J_test)


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
        emb_true, cond_info, freeze_options, fp_exp_domain, run_id, group_id=0):
    """Simulate random selection progress for a trial configuration.

    Record:
        n_trial, loss, r^2
    """
    # Define filepaths.
    fp_data_run = fp_exp_domain / Path('{0:s}_{1:s}_data.p'.format(cond_info['prefix'], run_id))
    fp_obs = fp_exp_domain / Path('{0:s}_{1:s}_obs.hdf5'.format(cond_info['prefix'], run_id))
    fp_emb_inf = fp_exp_domain / Path('{0:s}_{1:s}_emb_inf.hdf5'.format(cond_info['prefix'], run_id))

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
        emb_true, cond_info, freeze_options, fp_exp_domain, run_id):
    """Simulate active selection progress for a trial configuration.

    Record:
        n_trial, loss, r^2
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


def simulate_run_random_exp3(
        emb_true, cond_info, freeze_options, fp_exp_domain, run_id, obs_novice):
    """Simulate random selection progress for a trial configuration.

    Record:
        n_trial, loss, r^2
    """
    # Define filepaths.
    fp_data_run = fp_exp_domain / Path('{0:s}_{1:s}_data.p'.format(cond_info['prefix'], run_id))
    fp_obs = fp_exp_domain / Path('{0:s}_{1:s}_obs.hdf5'.format(cond_info['prefix'], run_id))
    fp_emb_inf = fp_exp_domain / Path('{0:s}_{1:s}_emb_inf.hdf5'.format(cond_info['prefix'], run_id))

    # Define expert agent based on true embedding.
    agent_expert = Agent(emb_true, group_id=1)

    # Expert similarity matrix associated with true embedding.
    
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
            (obs_novice, obs_expert.subset(include_idx))
        )

        loss[i_round] = emb_inferred.fit(
            obs_round, n_restart=10, init_mode=init_mode
        )  # TODO n_retart=50
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
        # emb_inferred.save(fp_emb_inf)  # TODO remove

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
    model_list = ['Exponential', 'HeavyTailed', 'StudentsT']
    pretty_list = ['Exponential', 'Heavy-Tailed', 'Student-t']
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
    [t, p] = stats.ttest_ind(x1, x2)
    df = len(x1) - 1
    print(
        '{0}:{1} | t({2}) = {3:.2f}, p = {4:.2f}'.format(
            m1, m2, df, t, p
        )
    )


def visualize_exp_2(data_r2c1, data_r8c2, data_a8c2, fp_figure):
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
        'verticalalignment': 'top',
        'horizontalalignment': 'left'
    }

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    i_cond = 2
    plot_exp2_condition(ax, data_r2c1, c_line[i_cond], c_env[i_cond], c_scatter[i_cond], fontdict)

    i_cond = 0
    plot_exp2_condition(ax, data_r8c2, c_line[i_cond], c_env[i_cond], c_scatter[i_cond], fontdict)

    i_cond = 1
    plot_exp2_condition(ax, data_a8c2, c_line[i_cond], c_env[i_cond], c_scatter[i_cond], fontdict)

    ax.set_ylim(bottom=0., top=1.)
    ax.set_xlabel('Total Worker Hours')
    ax.set_ylabel(r'$r^2$ Similarity')
    ax.legend()
    plt.tight_layout()

    if fp_figure is None:
        plt.show()
    else:
        plt.savefig(
            fp_figure.absolute().as_posix(), format='pdf',
            bbox_inches="tight", dpi=100)


def plot_exp2_condition(
        ax, data, c_line, c_env, c_scatter, fontdict, rsquared_crit=.9):
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
    
    r_squared_mask = ma.array(results['r_squared'], mask=np.logical_not(is_valid))
    
    r_squared_mean_avg = np.mean(r_squared_mask, axis=1)
    r_squared_mean_min = np.min(r_squared_mask, axis=1)
    r_squared_mean_max = np.max(r_squared_mask, axis=1)

    ax.semilogx(
        time_factor * n_trial_avg, r_squared_mean_avg, '-', color=c_line,
        label='{0:s}'.format(legend_name)
        # label='{0:s} ({1:d})'.format(legend_name, n_run)
    )
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
