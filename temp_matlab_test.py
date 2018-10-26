import copy
import itertools

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
import h5py

from psiz.trials import Docket
from psiz.models import Exponential
from psiz.simulate import Agent
from psiz.generator import ActiveGenerator
from psiz.utils import similarity_matrix


def main():
    """Compare hueristic and exhaustive search method for trial selection."""
    # Load.
    fp_matlab = '/Users/bdroads/Projects/psiz-app/matlab_version.mat'
    f = h5py.File(fp_matlab, 'r')
    rho = f['rho'][()][0][0]
    tau = f['tau'][()][0][0]
    beta = f['beta'][()][0][0]
    gamma = f['gamma'][()][0][0]
    z = np.transpose(f['z'][()])
    samples = dict(z=f['samples'][()])
    attention = np.transpose(f['attention'][()])

    # Instantiate.
    n_stimuli = z.shape[0]
    n_dim = z.shape[1]
    n_group = 1
    emb = Exponential(n_stimuli, n_dim=n_dim, n_group=n_group)
    emb.theta['rho']['value'] = rho
    emb.theta['tau']['value'] = tau
    emb.theta['gamma']['value'] = gamma
    emb.theta['beta']['value'] = beta
    emb.phi['phi_1']['value'] = attention
    emb.z['value'] = z

    n_reference = 8
    n_select = 2
    config_list = pd.DataFrame({
        'n_reference': np.array([n_reference], dtype=np.int32),
        'n_select': np.array([n_select], dtype=np.int32),
        'is_ranked': [True],
        'n_outcome': np.array([56], dtype=np.int32)  # TODO 2
    })
    gen = ActiveGenerator(config_list=config_list, n_neighbor=12)

    # Exhaustive search.
    stimulus_set = np.array([
        [360, 198, 248, 299, 56, 323, 69, 209, 188],
        [55, 198, 248, 299, 56, 323, 69, 209, 188],
        [376, 198, 248, 299, 56, 323, 69, 209, 188]
    ], dtype=np.int32) - 1
    n_candidate = stimulus_set.shape[0]
    candidate_docket = Docket(
        stimulus_set, n_select * np.ones(n_candidate, dtype=np.int32)
    )
    # ig = gen._information_gain(emb, samples, candidate_docket)
    n_trial = 3
    docket, ig = gen.generate(n_trial, emb, samples, n_query=3, verbose=0)
    print('here')

if __name__ == "__main__":
    main()
