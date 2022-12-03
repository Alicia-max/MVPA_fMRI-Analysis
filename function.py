import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA



# for 5-fold 9-subject CV (Alex's suggestion)
def make_chunks_per_9subjects(subject):
    chunks = np.ravel([[i]*30 for i in range(subject)])
    chunks = chunks % 5
    return chunks


# for leave-one-run-out across all subjects (2nd item in the email)
def make_chunks_per_run(subject):
    chunks = []
    for _ in range(subject): 
        for i in range(5): # Because 5 runs
            chunks.append(i*np.ones((6,), dtype=int)) # Because 6 conditions
    chunks = np.concatenate(chunks)
    return chunks

def accuracy(y,y_pred):
    """
    Computes the accuracy as the average of the correct predictions and returns it
    Input : 
        - y : the true prediction 
        - y_pred : model's prediction
    """
    score=0
    for idx, val in enumerate(y): 
        if val==y_pred[idx] : 
            score+=1
            
    return score/len(y)
