import os
import copy
import itertools
from pathlib import Path

import numpy as np
from numpy import ma
from scipy.stats import sem
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pickle
from psiz import models
from psiz import visualize
from psiz import trials
def main():
    fp_emb = '/home/brett/Projects/psiz-app.git/results/exp_2/birds/a8c2/a8c2_913_emb_inf.hdf5'
    fp_obs = '/home/brett/Projects/psiz-app.git/results/exp_2/birds/a8c2/a8c2_913_obs.hdf5'
    
    obs = trials.load_trials(fp_obs)

    emb = models.load_embedding(fp_emb)
    visualize.visualize_embedding_static(emb.z['value'])

if __name__ == "__main__":
    main()