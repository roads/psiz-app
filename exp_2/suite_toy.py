"""Experiment 2_t: Compare trial configurations and selection policies.

Since this experiment is computationally intensive, intermediate
results are saved to disk and loaded as needed.
"""

import os
import copy
import itertools
from pathlib import Path

import numpy as np
from scipy.stats import sem
import matplotlib
import matplotlib.pyplot as plt
import pickle
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
    # TODO rerun 2 choose 1
    # Settings.
    freeze_options = {'theta': {'rho': 2, 'beta': 10}}
    # seed_list = [913, 192, 785, 891, 841]
    seed_list = [913]

    # Filepaths.
    fp_emb_true = results_path / Path('emb_true_3d.hdf5')
    fp_data_r2c1 = results_path / Path('data_r2c1.p')
    fp_data_r8c2 = results_path / Path('data_r8c2.p')
    fp_data_a8c2 = results_path / Path('data_a8c2.p')
    fp_figure_embedding = results_path / Path('emb.pdf')
    fp_figure_exp2a = results_path / Path('exp2a.pdf')
    fp_figure_exp2b = results_path / Path('exp2b.pdf')

    time_s_2c1 = 3.06  # TODO compute from obs
    time_s_8c2 = 8.98  # TODO compute from obs

    # Define experiment conditions.
    cond_info_r2c1 = {
        'name': 'Random 2-choose-1',
        'selection_policy': 'random',
        'n_reference': 2,
        'n_select': 1,
        'n_trial_initial': 100,
        'n_trial_total': 4100,
        'n_trial_per_round': 500,
        'time_s_per_trial': time_s_2c1
    }
    cond_info_r8c2 = {
        'name': 'Random 8-choose-2',
        'selection_policy': 'random',
        'n_reference': 8,
        'n_select': 2,
        'n_trial_initial':  100,
        'n_trial_total': 5100,
        'n_trial_per_round': 250,
        'time_s_per_trial': time_s_8c2
    }
    cond_info_a8c2 = {
        'name': 'Active 8-choose-2',
        'selection_policy': 'active',
        'n_reference': 8,
        'n_select': 2,
        'n_trial_initial':  100,
        'n_trial_total': 10100,
        'n_trial_per_round': 500,
        'time_s_per_trial': time_s_8c2,
        'n_query': 25,
    }

    # Experiment 2 setup: Synthetic grouth truth.
    emb_true = ground_truth()
    # sim_mat = similarity_matrix(emb_true.similarity, emb_true.z['value'])
    # idx_upper = np.triu_indices(emb_true.n_stimuli, 1)
    # plt.hist(sim_mat[idx_upper])
    # plt.show()

    # simulate_multiple_runs(
    #     seed_list, emb_true, cond_info_r2c1, freeze_options, fp_data_r2c1)

    # simulate_multiple_runs(
    #     seed_list, emb_true, cond_info_r8c2, freeze_options, fp_data_r8c2)

    # simulate_multiple_runs(
    #     seed_list, emb_true, cond_info_a8c2, freeze_options, fp_data_a8c2)

    # Visualize Experiment 2 results.
    data_2c1_rand = pickle.load(open(fp_data_r2c1, 'rb'))
    # data_8c2_rand = pickle.load(open(fp_data_r8c2, 'rb'))
    plot_exp2([data_2c1_rand], fp_figure_exp2a)


def ground_truth():
    """Return a ground truth embedding."""
    np.random.seed(123)
    n_stimuli = 25  # 16
    n_dim = 2
    # Create embeddingp points arranged on a grid.
    x, y = np.meshgrid([-.15, -.05, .05, .15, .25], [-.15, -.05, .05, .15, .25])
    x = np.expand_dims(x.flatten(), axis=1)
    y = np.expand_dims(y.flatten(), axis=1)
    z = np.hstack((x, y))
    # Add some Gaussian noise to the embedding points.
    mean = np.zeros((n_dim))
    cov = .01 * np.identity(n_dim)
    z_noise = .1 * np.random.multivariate_normal(mean, cov, (n_stimuli))
    z = z + z_noise
    # Create embedding model.
    n_group = 1
    model = Exponential(n_stimuli, n_dim=n_dim, n_group=n_group)
    freeze_options = {
        'z': z,
        'theta': {
            'rho': 2,
            'tau': 1,
            'beta': 10,
            'gamma': 0.001
        }
    }
    model.freeze(freeze_options)

    # sim_mat = similarity_matrix(model.similarity, z)
    # idx_upper = np.triu_indices(model.n_stimuli, 1)
    # plt.hist(sim_mat[idx_upper])
    # plt.show()

    return model


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
        results_run = simulate_run(emb_true, cond_info, freeze_options)
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


def simulate_run(emb_true, cond_info, freeze_options):
    """Simulate a single run."""
    if cond_info['selection_policy'] is 'random':
        results_run = simulate_run_random(
            emb_true, cond_info, freeze_options)
    elif cond_info['selection_policy'] is 'active':
        results_run = simulate_run_active(
            emb_true, cond_info, freeze_options)
    else:
        raise ValueError(
            'The `selection_policy` must be either "random" or "active".'
        )
    return results_run


def simulate_run_random(emb_true, cond_info, freeze_options):
    """Simulate random selection progress for a trial configuration.

    Record:
        n_trial, loss, R^2
    """
    n_stimuli = emb_true.n_stimuli
    n_dim = emb_true.n_dim

    # Settings
    n_sample = 1000
    n_burn = 100
    thin_step = 3
    fp_snapshot_figure = Path('/Users/bdroads/Projects/psiz-app/results_toy/snap_')

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

    # Multiple samples per trial (x3). TODO
    # include_idx = np.arange(0, int(docket.n_trial/3))
    # include_idx = np.expand_dims(include_idx, axis=1)
    # include_idx = np.hstack((include_idx, include_idx, include_idx))
    # include_idx = include_idx.flatten()
    # docket = docket.subset(include_idx)

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
    # pearson = np.empty((n_round))
    # mse = np.empty((n_round))
    loss = np.empty((n_round))
    for i_round in range(n_round):
        include_idx = np.arange(0, n_trial[i_round])
        obs_round = obs.subset(include_idx)

        # Initialize embedding.
        emb_inferred = Exponential(n_stimuli, n_dim)
        emb_inferred.freeze(freeze_options)
        # Infer embedding with cold restarts.
        emb_inferred.set_log(True, delete_prev=True)  # TODO
        loss[i_round] = emb_inferred.fit(
            obs_round, n_restart=40, init_mode='cold')
        # Collect posterior samples
        samples = emb_inferred.posterior_samples(
            obs_round, n_sample, n_burn, thin_step)
        z_samp = samples['z']
        # z_central = np.median(z_samp, axis=2)
        z_samp = np.transpose(z_samp, axes=[2, 0, 1])
        z_samp = np.reshape(z_samp, (n_sample * n_stimuli, n_dim))

        # Compare the inferred model with ground truth by comparing the
        # similarity matrices implied by each model.
        simmat_infer = similarity_matrix(
            emb_inferred.similarity, emb_inferred.z['value'])
        r_squared[i_round] = matrix_comparison(
            simmat_infer, simmat_true, score='r2')
        # pearson[i_round] = matrix_comparison(
        #     simmat_infer, simmat_true, score='pearson')
        # mse[i_round] = matrix_comparison(
        #     simmat_infer, simmat_true, score='mse')
        # print(
        #     'Round {0} ({1} trials) | Loss: {2:.2f} '
        #     '| R^2: {3:.2f} | pearson: {4:.2f}| mse: {5:.3f}'.format(
        #         i_round, n_trial[i_round], loss[i_round], r_squared[i_round],
        #         pearson[i_round], mse[i_round]
        #     )
        # )
        print(
            'Round {0} ({1} trials) | Loss: {2:.2f} '
            '| R^2: {3:.2f}'.format(
                i_round, n_trial[i_round], loss[i_round], r_squared[i_round]
            )
        )
        save_snapshot(
            emb_true, emb_inferred, z_samp, n_sample,
            fp_snapshot_figure.absolute().as_posix() + str(i_round) + '.pdf',
            r_squared[i_round])

    results = {
        'n_trial': np.expand_dims(n_trial, axis=1),
        'loss': np.expand_dims(loss, axis=1),
        'r_squared': np.expand_dims(r_squared, axis=1)
    }
    return results


def simulate_run_active(emb_true, cond_info, freeze_options):
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
    n_trial = np.arange(
        cond_info['n_trial_initial'],
        cond_info['n_trial_total'] + 1,
        cond_info['n_trial_per_round']
    )
    n_round = len(n_trial)
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
    loss[i_round] = emb_inferred.fit(
        obs, n_restart=40, init_mode='cold', verbose=0)

    active_gen = ActiveGenerator()
    # Infer independent models with increasing amounts of data.
    for i_round in np.arange(1, n_round + 1):
        # Select trials based on expected IG.
        samples = emb_inferred.posterior_samples(obs, n_sample=1000, n_burn=10)
        active_docket, _ = active_gen.generate(
            cond_info['n_trial_per_round'], emb_inferred, samples,
            n_query=cond_info['n_query'])

        # Simulate observations.
        new_obs = agent.simulate(active_docket)
        obs = trials.stack([obs, new_obs])

        # Initialize embedding.
        emb_inferred = Exponential(emb_true.n_stimuli, emb_true.n_dim)
        emb_inferred.freeze(freeze_options)
        # Infer embedding with cold restarts.
        loss[i_round] = emb_inferred.fit(
            obs, n_restart=20, init_mode='cold', verbose=0)
        # Compare the inferred model with ground truth by comparing the
        # similarity matrices implied by each model.
        simmat_infer = similarity_matrix(
            emb_inferred.similarity, emb_inferred.z['value'])
        r_squared[i_round] = matrix_comparison(simmat_infer, simmat_true)
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


def save_snapshot(
        emb_true, emb_inferred, z_samp, n_sample, fp_figure, r_squared):
    """Save a snapshot of the run for offline inspection."""
    z_true = emb_true.z['value']
    z_central = emb_inferred.z['value']
    lims_x = [-.4, .5]
    lims_y = [-.4, .5]

    cmap = matplotlib.cm.get_cmap('jet')
    norm = matplotlib.colors.Normalize(vmin=0., vmax=emb_true.n_stimuli)
    color_array = cmap(norm(range(emb_true.n_stimuli)))
    color_array_samp = np.matlib.repmat(color_array, n_sample, 1)

    fig = plt.figure(figsize=(5.5, 2), dpi=200)

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.scatter(
        z_true[:, 0], z_true[:, 1], s=15, c=color_array, marker='o')
    ax1.set_title('Ground Truth')
    ax1.set_aspect('equal')
    ax1.set_xlim(lims_x[0], lims_x[1])
    ax1.set_xticks([])
    ax1.set_ylim(lims_y[0], lims_y[1])
    ax1.set_yticks([])

    ax2 = fig.add_subplot(1, 3, 2)
    scat2 = ax2.scatter(
        z_central[:, 0], z_central[:, 1],
        s=15, c=color_array, marker='X')
    ax2.set_title('Point Est. R^2={0:.2f}'.format(r_squared))
    ax2.set_aspect('equal')
    ax2.set_xlim(lims_x[0], lims_x[1])
    ax2.set_xticks([])
    ax2.set_ylim(lims_y[0], lims_y[1])
    ax2.set_yticks([])

    ax3 = fig.add_subplot(1, 3, 3)
    scat3 = ax3.scatter(
        z_samp[:, 0], z_samp[:, 1],
        s=5, c=color_array_samp, alpha=.01, edgecolors='none')
    ax3.set_title('Posterior Est.')
    ax3.set_aspect('equal')
    ax3.set_xlim(lims_x[0], lims_x[1])
    ax3.set_xticks([])
    ax3.set_ylim(lims_y[0], lims_y[1])
    ax3.set_yticks([])

    plt.savefig(
        fp_figure, format='pdf',
        bbox_inches="tight", dpi=200)


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
    results_path = Path('/Users/bdroads/Projects/psiz-app/results_toy')
    experiment_2(results_path)
