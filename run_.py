#Sklearn Library

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import permutation_test_score
from sklearn.metrics import f1_score

#Plot Library
import matplotlib.pyplot as plt

#for warnings
import sys
import os, warnings


def baseline_run(classifiers, chunks, train_samples, train_labels, dic) : 
    """
    Run a baseline model for each classifier and for each cross fold strategy (random 5CV, Leave one out per run or per subject)
    
    Inputs : 
    
    - classifiers : dictionnary containing classifiers referenced by their name
    - chunks : dictionnary with groups used for the Leave one out cross-validation strategy
    - train_samples : Data Matrix, containing vector of masked voxels for each run and for each subject at each time point
    - train_labels : vector of observed conditions for the data Matrix
    - dic : dictionnary filled with the computed accuracy
    """
    # Remove the convergence warning due to the imposed number of iterations
    if not sys.warnoptions : 
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARGNINGS"]=('ignore::ConvergenceWarning')
        
        # Iteration accros the given model 
        for name, clf in classifiers.items() : 
            
            print('\n'+ name + ', 5fold_CV :')
            
            # Initialisation of the estimator and Evaluation of the score by 5 cross-validation
            pipeline = Pipeline([('scale', StandardScaler()), (name, clf)])
            cv_scores=cross_val_score(pipeline, X=train_samples, y=train_labels,
                                      cv=5,n_jobs=-1,verbose=1)
            
             #Add mean score to the given dictionnary
            dic[name+'_5FCV'] = cv_scores.mean()
            
            #print score per fold & the mean of it
            print('Average accuracy = %.02f \n' % (cv_scores.mean()))
            print('Average Accuracy std = %.02f \n'% (cv_scores.std()))
            print('Accuracy per fold:', cv_scores, sep='\n')
            
            #Iteration across the groups to evaluate
            for key, chunk in chunks.items() : 
                
                print('\n'+ name + ','+ key + ' :' )
                
                # Initialisation of the estimator and Evaluation of the score by LeaveOneOut CV 
                cv_scores = cross_val_score(pipeline, X=train_samples, y=train_labels, 
                                            groups=chunk, cv=LeaveOneGroupOut(),n_jobs=-1, verbose=1)
                
                #Add mean score to the given dictionnary
                dic[name+'_'+key]= cv_scores.mean()
                
                #print score per fold & the mean of it
                print('Average accuracy = %.02f \n' % (cv_scores.mean()))
                print('Average Accuracy std = %.02f \n'% (cv_scores.std()))
                
def test_baseline(X_train, X_test, y_train, y_test, clf_dic):
    warnings.simplefilter("ignore")

    for name, clf in clf_dic.items():
        scaler = StandardScaler()
        std_samples_train =scaler.fit_transform(X_train)
        std_sample_test=scaler.transform(X_test)
        clf.fit(std_samples_train, y_train)
        y_pred= clf.predict(std_sample_test)
        acc=accuracy_score(y_test, y_pred)
        f1=f1_score(y_test, y_pred, average=None)
        print('Test accuracy Score for ',name, ' : ',  acc, '\n')  
        print('F1 score for', name, ':' , f1)
        cm=confusion_matrix(y_test, y_pred)
        disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        disp.plot()
        plt.xticks(rotation=60)
        plt.show()
        


def tuned_run(pipeline, grid,chunks, train_samples, train_labels, type_, dic, name_clf) :
    """
    Exhaustive search over specified parameter values for an estimator using GridSearchCV 
    with 3 different CV strategy (5 CV, OneLeaveOut per run and per subject).
    
    Inputs : 
    - pipeline : pipleine to implement the estimator
    - grid : Dictionary with parameters names (str) as keys and lists of parameter settings for the hyperparameters
    - chunks : dictionnary with cv strategy name (str) as key and the the groups used for the Leave one out cross-validation strategy
    - train_samples : Features Matrix
    - train_labels : vector of observed conditions for the data Matrix
    - type_ : string specifying the type of operation (for filling the dictionary)
    - dic : dictionnary filled with the computed accuracy
    - name_clf : string specifying the name of the classifier (for filling the dictionary)
    """
    
    # Remove the convergence warning due to the imposed number of iterations
    warnings.simplefilter("ignore")
    
    # 5 CV strategy
    print('\n'+ name_clf + ', 5fold CV : \n')

    # GridSearch initialisation
    grid_result = GridSearchCV(pipeline, param_grid = grid, scoring = 'accuracy',
                               verbose=1, n_jobs=-1, cv = 5)
    # Fit
    grid_result.fit(train_samples, train_labels)
    
    # Print Scores  & dic filling  
    print('Mean test score :' , grid_result.cv_results_['mean_test_score'], '\n')
    print('Test score std  :', grid_result.cv_results_['std_test_score'], '\n')
    print('Best Score: ', grid_result.best_score_) 
    print('Best Params: ', grid_result.best_params_) 
    dic[name_clf+'_5FCV_'+type_]=grid_result.best_score_
    
    #LeaveOneOut strategy
    for key, chunk in chunks.items():
        print('\n'+name_clf+ ' , ' +  key +' : \n')
        
        # GridSearch initialisation
        grid_result = GridSearchCV(pipeline, param_grid = grid, scoring = 'accuracy', 
                                   verbose=1, n_jobs=-1, cv =LeaveOneGroupOut())
        
        # Fit
        grid_result.fit(train_samples, train_labels ,groups = chunk)
        
        # Print Scores & dic filling
        print('Mean test score :' , grid_result.cv_results_['mean_test_score'], '\n')
        print('Test score std  :', grid_result.cv_results_['std_test_score'], '\n')
        dic[name_clf+'_'+key+type_]= grid_result.best_score_
        print('Best Score: ', grid_result.best_score_) 
        print('Best Params: ', grid_result.best_params_) 
        
        
def evaluation_test (pipeline, X, y, cv_, groups=None):
    warnings.simplefilter("ignore")
    
    null_cv_scores=permutation_test_score(estimator = pipeline, 
                                          X=X,
                                          y=y,
                                          groups=groups,
                                          cv=cv_, 
                                          n_permutations=30,
                                          n_jobs=-1,
                                          verbose=1)
    print('Prediction accuracy : %0.2f'% null_cv_scores[0], '\n',
              'p-value: %0.04f'%(null_cv_scores[2]))