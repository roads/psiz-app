"""
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from scipy.stats import sem
import multiprocessing
from functools import partial

# from hieralb.core import Album

from psiz.models import Observations, Exponential, HeavyTailed, StudentsT
from psiz.dimensionality import suggest_dimensionality
import psiz.utils as ut

# Disables tensorflow warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ALBUM_NAME = 'birds-16'
ALBUM_NAME = 'rocks_Nosofsky_etal_2016'

CV_PATH = Path('model_comparison', 'cv_10_fold')

def main():
    """
    """
    # Settings
    n_fold = 10

    (obs, n_stimuli) = load_obs(ALBUM_NAME)
    
    # Instantiate the balanced k-fold cross-validation object.
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=723)

    # suite_plot(ALBUM_NAME)
    
    # Exponential family.
    filepath = CV_PATH / Path(ALBUM_NAME, 'Exponential')
    freeze_options = {}
    loss = embedding_cv(skf, obs, Exponential, n_stimuli, freeze_options)
    pickle.dump(loss, open(str(filepath / Path("loss.p")), "wb"))

    # Gaussian family.
    # filepath = CV_PATH / Path(ALBUM_NAME, 'Gaussian')
    # freeze_options = dict(rho=2., tau=2., gamma=0.)
    # loss = embedding_cv(skf, obs, Exponential, n_stimuli, freeze_options)
    # pickle.dump(loss, open(str(filepath / Path("loss.p")), "wb"))

    # Laplacian family.
    # filepath = CV_PATH / Path(ALBUM_NAME, 'Laplacian')
    # freeze_options = dict(rho=2., tau=1., gamma=0.)
    # loss = embedding_cv(skf, obs, Exponential, n_stimuli, freeze_options)
    # pickle.dump(loss, open(str(filepath / Path("loss.p")), "wb"))

    # Heavy-tailed family.
    filepath = CV_PATH / Path(ALBUM_NAME, 'HeavyTailed')
    freeze_options = {}
    loss = embedding_cv(skf, obs, HeavyTailed, n_stimuli, freeze_options)
    pickle.dump(loss, open(str(filepath / Path("loss.p")), "wb"))

    # Student-t family.
    filepath = CV_PATH / Path(ALBUM_NAME, 'StudentsT')
    freeze_options = {}
    loss = embedding_cv(skf, obs, StudentsT, n_stimuli, freeze_options)
    pickle.dump(loss, open(str(filepath / Path("loss.p")), "wb"))

def suite_plot(ALBUM_NAME):
    filename = 'model_comparison/model_comparison.pdf'

    model_list = ['Exponential', 'HeavyTailed', 'StudentsT'] # 'Gaussian', 'Laplacian',
    n_model = len(model_list)

    train_mu = np.empty(n_model)
    train_se = np.empty(n_model)
    test_mu = np.empty(n_model)
    test_se = np.empty(n_model)
    for i_model, model_name in enumerate(model_list):
        filepath = CV_PATH / Path(ALBUM_NAME, model_name)
        loss = pickle.load(open(str(filepath / Path("loss.p")), "rb" ))

        train_mu[i_model] = np.mean(loss['train']) 
        train_se[i_model] = sem(loss['train'])

        test_mu[i_model] = np.mean(loss['test'])
        test_se[i_model] = sem(loss['test'])

    ind = np.arange(n_model)
    
    # Determine the maximum and minimum y values in the figure.
    ymin = np.min(np.stack((train_mu - train_se, test_mu - test_se), axis=0))
    ymax = np.max(np.stack((train_mu + train_se, test_mu + test_se), axis=0))
    ydiff = ymax - ymin
    ypad = ydiff * .1

    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, train_mu, width, color='r', yerr=train_se)

    rects2 = ax.bar(ind + width, test_mu, width, color='b', yerr=test_se)

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Loss')
    ax.set_title('Cross-validation Loss (10-fold)')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(model_list)

    ax.legend((rects1[0], rects2[0]), ('Train', 'Test'), loc=4)
    axes = plt.gca()
    # axes.set_xlim([xmin,xmax])
    axes.set_ylim([ymin - ypad,ymax + ypad])

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, format='pdf', bbox_inches="tight", dpi=100)

def evaluate_fold(i_fold, split_list, obs, embedding_constructor, n_stimuli, freeze_options, verbose):
    # Settings.
    n_restart_dim = 20
    n_restart_fit = 40

    if verbose > 1:
        print('    Fold: ', i_fold)

    (train_index, test_index) = split_list[i_fold]

    n_group = len(np.unique(obs.group_id))

    # Train.
    obs_train = obs.subset(train_index)    
    # Select dimensionality.
    n_dim = suggest_dimensionality(obs_train, embedding_constructor, n_stimuli,
    n_restart=n_restart_dim)
    # Instantiate model.
    embedding_model = embedding_constructor(n_stimuli, n_dim, n_group)
    if len(freeze_options) > 0:
        embedding_model.freeze(**freeze_options)
    # Fit model using training data.
    J_train = embedding_model.fit(obs_train, n_restart=n_restart_fit)
    
    # Test.
    obs_test = obs.subset(test_index)
    J_test = embedding_model.evaluate(obs_test)

    return (J_train, J_test)

def embedding_cv(skf, obs, embedding_constructor, n_stimuli, freeze_options):
    """
    """
    # Cross-validation settings.
    verbose = 2
    n_fold = skf.get_n_splits()

    J_train = np.empty((n_fold))
    J_test = np.empty((n_fold))

    split_list = list(skf.split(obs.stimulus_set, obs.configuration_id))
    # loaded_fold = lambda i_fold: evaluate_fold(i_fold, split_list, displays, display_info, embedding_constructor, n_stimuli, freeze_options, verbose)
    loaded_fold = partial(evaluate_fold, split_list=split_list, obs=obs, 
        embedding_constructor=embedding_constructor, n_stimuli=n_stimuli, 
        freeze_options=freeze_options, verbose=verbose)
    
    fold_list = range(n_fold)
    # for i_fold in fold_list:
    #     (J_train[i_fold], J_test[i_fold]) = loaded_func(i_fold)
    with multiprocessing.Pool() as pool:
        results = pool.map(loaded_fold, fold_list)

    J_train = []
    J_test = []
    for i_fold in fold_list:
        J_train.append(results[i_fold][0])
        J_test.append(results[i_fold][1])

    return {'train': J_train, 'test': J_test}    

def load_obs(obspath):
    """
    """
    n_stimuli = np.asscalar(np.loadtxt(Path(obspath, 'n_stimuli.txt'), dtype='int'))
    
    # Load observations
    displays = pd.read_csv(Path(obspath, 'judged_displays.txt'), header=None, dtype=np.int32)
    displays = displays.as_matrix()
    display_info = pd.read_csv(Path(obspath, 'display_info.txt'))
    
    # Create observation object.
    n_selected = np.array(display_info.n_selected)
    obs = Observations(displays, n_selected)

    return (obs, n_stimuli)

if __name__ == "__main__":
    main()

# load('/Users/bdroads/Dropbox/MATLAB Projects/psychEmbed/applications/model_comparison/cv/expSDE/score.mat')
# expTestLoss = score.test(:,1);
# load('/Users/bdroads/Dropbox/MATLAB Projects/psychEmbed/applications/model_comparison/cv/hSDE/score.mat')
# hTestLoss = score.test(:,1);
# %%
# [h,p,ci,stats] = ttest(expTestLoss, hTestLoss);
# str = apa_paired_t_test(stats, p);
# fprintf('%s\n', str)
