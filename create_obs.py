"""Script to create Observation objects from text files."""

import numpy as np
import tensorflow as tf
from pathlib import Path
import pandas as pd
from psiz.trials import Observations

# APP_PATH = Path('rocks_Nosofsky_etal_2016')
# APP_PATH = Path('datasets', 'birds-12')
APP_PATH = Path('/Users', 'bdroads', 'Projects', 'psiz-app', 'datasets')
dataset = Path('birds-12')
# dataset = Path('birds-16')
# dataset = Path('lesions')
# dataset = Path('rocks_Nosofsky_etal_2016')


def main():
    """Create dataset files.

    Create observation hdf5 file.
    Create album_info.txt.
    """
    obs_filepath = APP_PATH / dataset / Path('obs.hdf5')

    # album_info = pd.read_csv(APP_PATH / dataset / Path('album_info.txt'))
    display_info = pd.read_csv(APP_PATH / dataset / Path('display_info.txt'))
    stimulus_set = pd.read_csv(
        APP_PATH / dataset / Path('judged_displays.txt'),
        header=None, dtype=np.int32)
    stimulus_set = stimulus_set.as_matrix()
    # stimulus_set = stimulus_set - 1 # subtract 1 for zero indexing
    unique_id_list = np.unique(stimulus_set)
    print('stimulus_set')
    print('  min: {0}'.format(np.min(stimulus_set)))
    print('  max: {0}'.format(np.max(stimulus_set)))
    print('  unique: {0}'.format(len(unique_id_list)))
    print('')

    n_select = np.array(display_info.n_select)
    print('n_select')
    print('  min: {0}'.format(np.min(n_select)))
    print('  max: {0}'.format(np.max(n_select)))
    print('')

    obs = Observations(stimulus_set, n_select=n_select)
    obs.save(obs_filepath)

if __name__ == "__main__":
    main()
