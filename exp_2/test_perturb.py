"""Tempory script to test perturbation of embedding."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import mixture
import matplotlib.pyplot as plt
import matplotlib


def main():
    """Main."""
    n_dim = 2
    n_stimuli = 16
    results_path = Path('/Users/bdroads/Projects/psiz-app/results')
    fp_figure = results_path / Path('temp_pertub.pdf')
    lims_x = [-.4, .4]
    lims_y = [-.4, .4]

    # Original.
    mu_orig = np.zeros((n_dim))
    cov_orig = .01 * np.identity(n_dim)
    z_orig = np.random.multivariate_normal(mu_orig, cov_orig, (n_stimuli))

    # Perturbed.
    gmm = mixture.GaussianMixture(
        n_components=1, covariance_type='spherical')
    gmm.fit(z_orig)
    mu = gmm.means_[0]
    cov = gmm.covariances_[0] * np.identity(n_dim)
    print('Fitted cov: {0:.4f}'.format(cov[0, 0]))
    # print(cov[0, 0])
    z_noise = np.random.multivariate_normal(mu, cov, (n_stimuli))
    z_pert = .8 * z_orig + .2 * z_noise

    cmap = matplotlib.cm.get_cmap('jet')
    norm = matplotlib.colors.Normalize(vmin=0., vmax=n_stimuli)
    color_array = cmap(norm(range(n_stimuli)))

    fig = plt.figure(figsize=(5.5, 2), dpi=200)

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(
        z_orig[:, 0], z_orig[:, 1], s=15, c=color_array, marker='o')
    ax1.set_title('Original')
    ax1.set_aspect('equal')
    ax1.set_xlim(lims_x[0], lims_x[1])
    ax1.set_xticks([])
    ax1.set_ylim(lims_y[0], lims_y[1])
    ax1.set_yticks([])

    ax2 = fig.add_subplot(1, 2, 2)
    scat2 = ax2.scatter(
        z_pert[:, 0], z_pert[:, 1],
        s=15, c=color_array, marker='o')
    ax2.set_title('Perturbed')
    ax2.set_aspect('equal')
    ax2.set_xlim(lims_x[0], lims_x[1])
    ax2.set_xticks([])
    ax2.set_ylim(lims_y[0], lims_y[1])
    ax2.set_yticks([])

    plt.tight_layout()
    plt.savefig(
        fp_figure.absolute().as_posix(), format='pdf',
        bbox_inches="tight", dpi=200)


if __name__ == "__main__":
    main()