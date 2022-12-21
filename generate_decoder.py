
from nilearn.decoding import Decoder
from sklearn.model_selection import LeaveOneGroupOut

from fonction import make_chunks_per_run, make_chunks_per_subjects

from dataset import Dataset
import pickle
import os
import sys

from nilearn.decoding import Decoder

import logging

def generate_decoder(datadir, cv_strategy, decoder, param_decoder, saving_dir, debug=False):
    '''
    TODO
    '''

    ## Loading data
    dataset = Dataset(datadir, debug)
    print(dataset.beta_maps)

    ## Setting up CV strategy
    if cv_strategy == 'per_run':
        cv = LeaveOneGroupOut()
        chunks = make_chunks_per_run(dataset.nb_subs_, dataset.nb_runs_per_sub_)
    elif cv_strategy == 'per_subs':
        cv = LeaveOneGroupOut()
        chunks = make_chunks_per_subjects(dataset.nb_subs_)
        print(chunks)
    else:
        logging.info('ERROR, {} cv not implemented for this method'.format(cv_strategy))
        sys.exit(-1)

    logging.info('Starting to fit decoder...')
    decoder = Decoder(
        estimator=decoder,
        mask=dataset.mask,
        cv=cv,
        param_grid=param_decoder,
        n_jobs=-1,
        verbose=1,
        standardize=True
    )

    decoder.fit(dataset.get_beta_maps(), dataset.get_labels(), groups = chunks)
    
    ## Here save the decoder in saving_dir
    logging.info('\n------------------------')
    logging.info('Decoder fitted with scores')
    logging.info(decoder.cv_scores_)
    logging.info('------------------------')

    pickle.dump(decoder, open(os.path.join(saving_dir, 'train_decoder.sav'), 'wb'))
    logging.info('Fitted decoder saved')