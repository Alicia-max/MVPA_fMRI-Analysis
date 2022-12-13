import numpy as np

# for 5-fold 9-subject CV (Alex's suggestion)
def make_chunks_per_9subjects () :
    chunks = np.ravel([[i]*30 for i in range(45)])
    chunks = chunks % 5
    return chunks


# for leave-one-run-out across all subjects (2nd item in the email)
def make_chunks_per_run() :
    chunks = []
    for _ in range(45): 
        for i in range(5): # Because 5 runs
            chunks.append(i*np.ones((6,), dtype=int)) # Because 6 conditions
    chunks = np.concatenate(chunks)
    return chunks

def split(samples,labels, nb_test) :
    nb_sub = 49
    test_samples = samples[0:nb_test*30,:]
    train_samples = samples[nb_test*30:nb_sub*30,:]
    test_labels = labels[0:nb_test*30]
    train_labels = labels[nb_test*30:nb_sub*30]
    
    print('Test sample shape', test_samples.shape, ', label size : ', test_labels.size)
    print('Train sample shape', train_samples.shape, ', label size : ', train_labels.size)
    
    return train_samples, test_samples, train_labels, test_labels
