import multiprocessing
from itertools import product
from functools import partial
from sklearn.model_selection import StratifiedKFold
import numpy as np

def evaluate_fold(i_fold, split_list, verbose):
    if verbose > 1:
            print('    Fold: ', i_fold)
    (train_idx, test_idx) = split_list[i_fold]
    # Do stuff

    J_train = 0
    J_test = 0
    return (J_train, J_test)

if __name__ == '__main__':
    n_fold = 3
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=723)

    displays = np.ones((10,3))
    display_type_id = np.ones((10))

    split_list = list(skf.split(displays, display_type_id))
    verbose = 2

    fold_list = range(n_fold)
    with multiprocessing.Pool() as pool:
        results = pool.map(partial(evaluate_fold, split_list=split_list, verbose=verbose), fold_list)
    print(results)