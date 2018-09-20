"""Experiment 3:  Selection comparison for one group.

Comparing random versus active selection for one group using simulations.

An initial embedding is inferred using real observations data. This
embedding is then treated as ground truth for the purpose of simulating
human behavior for two different display configurations.

Notes:
    For simiplicity, the dimensionality is not inferred at each step.
        Instead, the correct dimensionality is provided.
    Perform multiple runs.

"""

import os

import numpy as np
from psiz.models import Exponential, load_embedding
from psiz.simulate import Agent
from psiz.generator import RandomGenerator
from psiz import datasets
from psiz.utils import similarity_matrix, matrix_correlation


def experiment_3():
    emb_filepath = os.path.join(results_path, 'emb_true_3d.hdf5')

    # Load ground truth embedding.
    emb_true = load_embedding(emb_filepath)


def simulate_active_condition(emb_true, cond_info, freeze_options):
    """Simulate active selection progress for a trial configuration.

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


if __name__ == "__main__":
    results_path = '/Users/bdroads/Projects/psiz-app/results'
    experiment_3(results_path)
