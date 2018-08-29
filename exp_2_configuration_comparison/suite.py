"""Comparing different trial configurations using simulations.

An initial embedding is inferred using real observations data. This
embedding is then treated as ground truth for the purpose of simulating
human behavior for two different display configurations.
"""

import copy
import itertools

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from psiz.models import Exponential, load_embedding
from psiz.dimensionality import suggest_dimensionality
from psiz import datasets


def main():
    """Run experiment 2."""
    emb_filepath = 'emb_true.hdf5'

    emb_true = real_embedding()
    emb_true.save(emb_filepath)

    emb_true = load_embedding(emb_filepath)
    simulate_2_choose_1(emb_true)

    emb_true = load_embedding(emb_filepath)
    simulate_8_choose_2(emb_true)


def real_embedding():
    """Fit embedding model to real observations."""
    # Settings.
    dataset_name = 'birds-16'
    (obs, catalog) = datasets.load_dataset(dataset_name)
    # Determine dimensionality.
    freeze_options = {'theta': {'rho': 2}}
    n_dim = suggest_dimensionality(
        obs, Exponential, catalog.n_stimuli, freeze_options=freeze_options,
        verbose=1)
    # Determine embedding using all available observation data.
    emb_true = Exponential(catalog.n_stimuli, n_dim)
    emb_true.fit(obs, n_restart=20, verbose=2)
    return emb_true


def simulate_2_choose_1(emb_true):
    """Simulate progress for a 2-choose-1 trial configuration."""
    return None


def simulate_8_choose_2(emb_true):
    """Simulate progress for a 8-choose-2 trial configuration."""
    return None


if __name__ == "__main__":
    main()
