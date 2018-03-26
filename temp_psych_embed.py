import numpy as np
import tensorflow as tf
from pathlib import Path
import pandas as pd
from hieralb.core import Album

from psiz.models import Exponential, HeavyTailed, StudentsT
from psiz.dimensionality import suggest_dimensionality

# Disables tensorflow warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# tensorboard --logdir=/tmp/tensorflow_logs/embedding/
# TODO test with attention weights

DROPBOX_PATH = Path('/Users/bdroads/Dropbox')
APP_PATH = DROPBOX_PATH / Path('python_projects', 'psych_embed_app', 'birds_12')
ALBUM_PATH = DROPBOX_PATH / Path('exp-datasets', 'birds-12')

def main():
    """ MAIN
    Args:
      abc
    Returns:
      abc
    Raises:
      abc
    """

    # album = Album(ALBUM_PATH)
    # album_info = pd.read_csv(APP_PATH / Path('album_info.txt'))
    display_info = pd.read_csv(APP_PATH / Path('display_info.txt'))
    judged_displays = pd.read_csv(APP_PATH / Path('judged_displays.txt'), header=None, dtype=np.int32)
    judged_displays = judged_displays - 1 # subtract 1 for zero indexing
    judged_displays = judged_displays.as_matrix()

    # # HACK test 8 choose 2 only
    # locs = display_info.n_reference == 8
    # judged_displays = judged_displays[locs,:]
    # display_info = display_info[locs]

    unique_id_list = np.unique(judged_displays)
    unique_id_list_less = unique_id_list[unique_id_list >= 0]
    n_stimuli = len(unique_id_list_less)

    # Embed
    n_selected=np.array(display_info.n_selected)

    # n_dim = suggest_dimensionality(Exponential, n_stimuli, judged_displays, n_selected=n_selected, n_restart=2, verbose=2)
    # print('n_dim: ', n_dim)
    
    dimensionality = 3
    n_group = 1    
    embedding_model = Exponential(n_stimuli, dimensionality, n_group)
    # embedding_model = HeavyTailed(n_stimuli, dimensionality, n_group)
    # embedding_model = StudentsT(n_stimuli, dimensionality, n_group)
    embedding_model.set_log(do_log=True, delete_prev=True)
    loss = embedding_model.fit(judged_displays, n_selected=n_selected, n_restart=5, verbose=1)

    # embedding_model.reuse(True, .05)
    # loss = embedding_model.fit(judged_displays, n_dim, n_selected=n_selected, n_restart=2, verbose=1)
    # print('loss: ', loss)
    # print('rho: ', embedding_model.rho)
    # print('tau: ', embedding_model.tau)
    # print('beta: ', embedding_model.beta)
    # print('gamma: ', embedding_model.gamma)

if __name__ == "__main__":
    main()
