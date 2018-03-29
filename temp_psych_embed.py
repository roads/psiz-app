import numpy as np
import tensorflow as tf
from pathlib import Path
import pandas as pd

from psiz.models import Observations, Exponential, HeavyTailed, StudentsT
from psiz.dimensionality import suggest_dimensionality

# Disables tensorflow warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# tensorboard --logdir=/tmp/tensorflow_logs/embedding/
# TODO test with attention weights

# APP_PATH = Path('birds-12')
APP_PATH = Path('/home', 'brett', 'Projects', 'psiz-app.git', 'birds-12')

def main():
    """ MAIN
    Args:
      abc
    Returns:
      abc
    Raises:
      abc
    """

    # album_info = pd.read_csv(APP_PATH / Path('album_info.txt'))
    display_info = pd.read_csv(APP_PATH / Path('display_info.txt'))
    judged_displays = pd.read_csv(APP_PATH / Path('judged_displays.txt'), header=None, dtype=np.int32)
    judged_displays = judged_displays - 1 # subtract 1 for zero indexing
    judged_displays = judged_displays.as_matrix()
    n_selected=np.array(display_info.n_selected)

    obs = Observations(judged_displays, n_selected=n_selected)

    unique_id_list = np.unique(judged_displays)
    unique_id_list_less = unique_id_list[unique_id_list >= 0]
    n_stimuli = len(unique_id_list_less)
    n_group = 1
    n_dim = 3 
    
    # Embed    
    # n_dim = suggest_dimensionality(obs, Exponential, n_stimuli, n_restart=2, verbose=2)
    # print('n_dim: ', n_dim)
    
    embedding_model = Exponential(n_stimuli, n_dim, n_group)
    # embedding_model = HeavyTailed(n_stimuli, dimensionality, n_group)
    # embedding_model = StudentsT(n_stimuli, dimensionality, n_group)
    # embedding_model.set_log(do_log=True, delete_prev=True)
    loss = embedding_model.fit(obs, n_restart=4, verbose=1)

    # embedding_model.reuse(True, .05)
    # loss = embedding_model.fit(obs, n_restart=2, verbose=1)

    print('Loss:', loss)

if __name__ == "__main__":
    main()
