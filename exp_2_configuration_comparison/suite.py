"""Example comparing random and active selection.

This example using simulated behavior to illustrate the theoretical
advantage of using active selection over random trial selection. The
simulation is time intensive.

"""

import copy
import itertools

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from psiz.models import Exponential
from psiz.dimensionality import suggest_dimensionality
from psiz import datasets


def main():
    """Fit embedding model to observations."""
    # Settings.
    emb_fname = 'emb.hdf5'

    dataset_name = 'birds-16'
    (obs, catalog) = datasets.load_dataset(dataset_name)

    # Determine dimensionality.
    freeze_options = {'theta': {'rho': 2}}
    n_dim = suggest_dimensionality(
        obs, Exponential, catalog.n_stimuli, freeze_options=freeze_options,
        verbose=1)

    # Gradient decent solution.
    emb = Exponential(catalog.n_stimuli, n_dim)
    emb.fit(obs, n_restart=20, verbose=2)
    emb.save(emb_fname)

if __name__ == "__main__":
    main()
