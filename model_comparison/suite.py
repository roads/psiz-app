'''
'''
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from scipy.stats import sem

from hieralb.core import Album

from psiz.models import Exponential, HeavyTailed, StudentsT
from psiz.dimensionality import suggest_dimensionality
import psiz.utils as ut

# Disables tensorflow warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DROPBOX_PATH = Path('/Users/bdroads/Dropbox')
# DROPBOX_PATH = Path('C:\Users\Brett\Dropbox\Datasets')

ALBUM_NAME = 'birds-16'
APP_PATH = DROPBOX_PATH / Path('python_projects', 'psych_embed_app', ALBUM_NAME)
ALBUM_PATH = DROPBOX_PATH / Path('exp-datasets', ALBUM_NAME)
CV_PATH = DROPBOX_PATH / Path('python_projects', 'psych_embed_app', 'model_comparison', 'cv_10_fold')

def main():
    '''
    '''

    n_fold = 10

    album = Album(ALBUM_PATH)

    # Import judged displays.
    displays = pd.read_csv(APP_PATH / Path('judged_displays.txt'), header=None, dtype=np.int32)
    displays = displays - 1 # subtract 1 for zero indexing
    displays = displays.as_matrix()
    # Import corresponding display info.
    display_info = pd.read_csv(APP_PATH / Path('display_info.txt'))

    # Instantiate the balanced k-fold cross-validation object.
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=723)

    # suite_plot(ALBUM_NAME)

    # Exponential family. DONE
    # filepath = CV_PATH / Path(ALBUM_NAME, 'Exponential')
    # freeze_options = {}
    # loss = embedding_cv(skf, displays, display_info, Exponential, album.n_stimuli, freeze_options)
    # pickle.dump(loss, open(str(filepath / Path("loss.p")), "wb"))

    # Gaussian family.
    # filepath = CV_PATH / Path(ALBUM_NAME, 'Gaussian')
    # freeze_options = dict(rho=2., tau=2., gamma=0.)
    # loss = embedding_cv(skf, displays, display_info, Exponential, album.n_stimuli, freeze_options)
    # pickle.dump(loss, open(str(filepath / Path("loss.p")), "wb"))

    # Laplacian family.
    # filepath = CV_PATH / Path(ALBUM_NAME, 'Laplacian')
    # freeze_options = dict(rho=2., tau=1., gamma=0.)
    # loss = embedding_cv(skf, displays, display_info, Exponential, album.n_stimuli, freeze_options)
    # pickle.dump(loss, open(str(filepath / Path("loss.p")), "wb"))

    # Heavy-tailed family. DONE
    # filepath = CV_PATH / Path(ALBUM_NAME, 'HeavyTailed')
    # freeze_options = {}
    # loss = embedding_cv(skf, displays, display_info, HeavyTailed, album.n_stimuli, freeze_options)
    # pickle.dump(loss, open(str(filepath / Path("loss.p")), "wb"))

    # Student-t family.
    filepath = CV_PATH / Path(ALBUM_NAME, 'StudentsT')
    freeze_options = {}
    loss = embedding_cv(skf, displays, display_info, StudentsT, album.n_stimuli, freeze_options)
    pickle.dump(loss, open(str(filepath / Path("loss.p")), "wb"))

def suite_plot(ALBUM_NAME):
    filename = 'psych_embed_app/model_comparison/model_comparison.pdf'

    model_list = ['Exponential', 'HeavyTailed'] # StudentsT, 'Gaussian', 'Laplacian',
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

def embedding_cv(skf, displays, display_info, embedding_constructor, n_stimuli, freeze_options):
    '''
    '''
    # Cross-validation settings.
    verbose = 2
    n_fold = skf.get_n_splits()
    n_restart = 40
    
    n_display = displays.shape[0]
    # TODO group_id
    group_id = np.zeros((n_display))
    n_group = len(np.unique(group_id))

    # Unpackage observations
    n_reference = np.array(display_info.n_reference)
    n_selected = np.array(display_info.n_selected)
    is_ranked = np.array(display_info.is_ranked)
    assignment_id = np.array(display_info.assignment_id)

    # Infer the display type IDs.
    display_type_id = ut.generate_display_type_id(n_reference, n_selected, 
    is_ranked, group_id, assignment_id)
    
    J_train = np.empty((n_fold))
    J_test = np.empty((n_fold))
    i_fold = 0
    for train_index, test_index in skf.split(displays, display_type_id):
        if verbose > 1:
            print('    Fold: ', i_fold)

        # Train.
        displays_train = displays[train_index,:]
        n_selected_train = n_selected[train_index]
        is_ranked_train = is_ranked[train_index]
        group_id_train = group_id[train_index]
        # Select dimensionality.
        n_dim = suggest_dimensionality(embedding_constructor, n_stimuli, 
        displays_train, n_selected=n_selected_train, is_ranked=is_ranked_train,
        group_id=group_id_train, n_restart=20, verbose=0)
        # Instantiate model.
        embedding_model = embedding_constructor(n_stimuli, n_dim, n_group)
        if len(freeze_options) > 0:
            embedding_model.freeze(**freeze_options)
        # Fit model using training data.
        J_train[i_fold] = embedding_model.fit(displays_train, 
        n_selected=n_selected_train, is_ranked=is_ranked_train, 
        group_id=group_id_train, n_restart=n_restart, verbose=0)
        
        # Test.
        displays_test = displays[test_index,:]
        n_selected_test = n_selected[test_index]
        is_ranked_test = is_ranked[test_index]
        group_id_test = group_id[test_index]
        J_test[i_fold] = embedding_model.evaluate(displays_test, 
        n_selected=n_selected_test, is_ranked=is_ranked_test, 
        group_id=group_id_test)

        i_fold = i_fold + 1

    return {'train': J_train, 'test': J_test}    

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
