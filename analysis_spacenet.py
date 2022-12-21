
from dataset import Dataset
import pickle

import os
import sys
import numpy as np

import logging
from fonction import make_chunks_per_run, make_chunks_per_subjects, make_folds_from_chunks

from nilearn.decoding import SpaceNetClassifier

def _spacenet_custom_CV(decoder, dataset, folds):
    '''
    TODO
    '''
    accuracies = []
    for train_idx, validation_idx in folds:
        X_train, X_val, y_train, y_val = dataset.split_train_val(train_idx, validation_idx)
        decoder.fit(X_train, y_train)
        y_pred = decoder.predict(X_val)
        accuracies.append(np.mean(y_pred == y_val))

    print(accuracies)
    
    return {
        'trained_decoder': decoder,
        'accuracies': accuracies
    }


def analysis_spacenet(datadir, cv_strategy, penalty, param_spacenet, saving_dir, debug=False):
    '''
    TODO
    '''

    logging.info('Running spacenent with penalty {}'.format(penalty))

    ## Loading data and scaling it
    dataset = Dataset(datadir, debug)

    ## Setting up CV strategy
    if cv_strategy == 'per_run':
        chunks = make_chunks_per_run(dataset.nb_subs_, dataset.nb_runs_per_sub_)
    elif cv_strategy == 'per_subs':
        chunks = make_chunks_per_subjects(dataset.nb_subs_)
        print(chunks)
    else:
        logging.info('ERROR, {} cv not implemented for this method'.format(cv_strategy))
        sys.exit(-1)

    folds = make_folds_from_chunks(chunks)

    for alpha in param_spacenet['alphas']:
        decoder = SpaceNetClassifier(
            penalty=penalty,
            mask=dataset.mask,
            max_iter = param_spacenet['max_iter'][0],
            alphas=alpha,
            cv=1,
            n_jobs=1,
            standardize=True,
        )
        results = _spacenet_custom_CV(decoder, dataset, folds)
        logging.info('\n------------------------')
        logging.info('Scores for alpha= {}'.format(str(alpha)))
        logging.info(results['accuracies'])
        logging.info('------------------------')

        pickle.dump(results, open(os.path.join(saving_dir, 'trained_spacenet_{}.sav'.format(alpha)), 'wb'))