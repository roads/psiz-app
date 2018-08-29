"""Script to create Observation objects from text files."""

import numpy as np
from pathlib import Path
import pandas as pd
from psiz.trials import Observations
from psiz.datasets import Catalog

ALBUM_PATH = Path('/Users', 'bdroads', 'Dropbox', 'exp-datasets')
APP_PATH = Path('/Users', 'bdroads', 'Projects', 'psiz-app', 'datasets')
HOST_PATH = Path(
    '/Users', 'bdroads', 'Websites', 'homepage', 'homepage', 'static',
    'psiz', 'datasets')


def main():
    """Create dataset files for server.

    1. Create catalog hdf5 file.
    2. Create observation hdf5 file.

    """
    dataset_list = [
        Path('birds-12'),
        Path('birds-16'),
        Path('lesions'),
        Path('rocks_Nosofsky_etal_2016')
    ]

    for dataset in dataset_list:
        catalog_filepath = HOST_PATH / dataset / Path('catalog.hdf5')
        obs_filepath = HOST_PATH / dataset / Path('obs.hdf5')

        # Create Catalog object.
        # album_info = pd.read_csv(APP_PATH / dataset / Path('album_info.txt'))
        album_path = ALBUM_PATH / dataset
        image_class0 = pd.read_csv(
            album_path / Path("image_class0.txt"), header=None,
            names=('stimulus_id', 'leaf_class_id'), delimiter=' ')
        images = pd.read_csv(
            album_path / Path("images.txt"), header=None,
            names=('stimulus_id', 'path'), delimiter=' ')
        stimuli = pd.merge(image_class0, images, on='stimulus_id')

        catalog = Catalog(
            stimuli.stimulus_id.values - min(stimuli.stimulus_id.values),
            stimuli.path.values
        )

        # Create Observations object.
        display_info = pd.read_csv(
            APP_PATH / dataset / Path('display_info.txt'))
        stimulus_set = pd.read_csv(
            APP_PATH / dataset / Path('judged_displays.txt'),
            header=None, dtype=np.int32)
        stimulus_set = stimulus_set.as_matrix()
        # stimulus_set = stimulus_set - 1 # subtract 1 for zero indexing
        # unique_id_list = np.unique(stimulus_set)
        # print('stimulus_set')
        # print('  min: {0}'.format(np.min(stimulus_set)))
        # print('  max: {0}'.format(np.max(stimulus_set)))
        # print('  unique: {0}'.format(len(unique_id_list)))
        # print('')

        n_select = np.array(display_info.n_select)
        # print('n_select')
        # print('  min: {0}'.format(np.min(n_select)))
        # print('  max: {0}'.format(np.max(n_select)))
        # print('')

        obs = Observations(stimulus_set, n_select=n_select)

        # Save
        catalog.save(catalog_filepath)
        obs.save(obs_filepath)

if __name__ == "__main__":
    main()
