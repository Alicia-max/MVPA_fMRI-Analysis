## baseline analysis
import sys
import logging

from dataset import Dataset
from sklearn.model_selection import LeaveOneGroupOut, ParameterGrid, cross_val_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from fonction import make_chunks_per_run, make_chunks_per_subjects

def _select_model(modelstr, params):
    '''
    TODO
    '''
    if modelstr == 'logistic':
        return LogisticRegression(**params)
    elif modelstr == 'ridge':
        return RidgeClassifier(**params)
    elif modelstr == 'linearsvc':
        return LinearSVC(**params)
    else:
        logging.info('Model {} not implemented. Aborting.'.format(modelstr))
        sys.exit(-1)

def _analysis_baseline_per_model(X, y, model, cv, chunks):
    '''
    TODO
    '''    
    scores = cross_val_score(
        estimator=model,
        X=X,
        y=y,
        cv=cv,
        groups=chunks,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    return scores

def analysis_baseline(datadir, cv_strategy, models, params_models, debug=False):
    '''
    TODO
    '''
    ## Loading data
    dataset = Dataset(datadir, debug)
    X, y = dataset.get_samples(), dataset.get_labels()
    
    ## Setting up CV strategy
    if cv_strategy == 'per_run':
        cv = LeaveOneGroupOut()
        chunks = make_chunks_per_run(dataset.nb_subs_, dataset.nb_runs_per_sub_)
    elif cv_strategy == 'per_subs':
        cv = LeaveOneGroupOut()
        chunks = make_chunks_per_subjects(dataset.nb_subs_)
        print(chunks)
    elif cv_strategy == 'random':
        cv = 5
        chunks = None
    else:
        logging.info('ERROR, {} cv not implemented for this method'.format(cv_strategy))
        sys.exit(-1)
    

    for modelstr, params in zip(models, params_models):
        param_grid = ParameterGrid(param_grid=params)
        for p in param_grid:
            logging.info('\n------------------------')
            logging.info('Scores for the model {}'.format(str(modelstr)))
            try:
                model = _select_model(modelstr, p)
                pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
                accuracies = _analysis_baseline_per_model(X, y, pipe, cv, chunks)
                logging.info('Accuracy: {} +/- {}'.format(accuracies.mean(), accuracies.std()))
            except:
                logging.info('ERROR, could not perform CV')
            logging.info('With parameters: {}'.format(p))
            logging.info('------------------------')
