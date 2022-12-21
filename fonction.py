import numpy as np

###############################
### CV helpers
###############################

def make_chunks_per_subjects(n_subs, n_maps_per_subs = 30, n = 7):
    '''
    Assumes same number of features per subject.
    '''
    chunks = np.ravel([[i]*n_maps_per_subs for i in range(n_subs)])
    chunks = chunks % n
    return chunks

# for leave-one-run-out across all subjects (2nd item in the email)
def make_chunks_per_run(n_subs, n_runs_per_sub) :
    '''
    TODO
    '''
    chunks = []
    for _ in range(n_subs): 
        for i in range(n_runs_per_sub): # Because 5 runs
            chunks.append(i*np.ones((6,), dtype=int)) # Because 6 conditions
    chunks = np.concatenate(chunks)
    return chunks

def make_folds_from_chunks(chunks):
    '''
    TODO
    '''
    idx = np.arange(len(chunks))

    folds = []
    for i in np.unique(chunks):
        train_idx = idx[chunks != i]
        validation_idx = idx[chunks == i]
        folds.append((train_idx, validation_idx))
    return folds


###############################
### Misc
###############################