from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import permutation_test_score

import sys
import os, warnings


def baseline_run(classifiers, chunks, train_samples, train_labels, dic) : 
    if not sys.warnoptions : 
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARGNINGS"]=('ignore::ConvergenceWarning')
        for name, clf in classifiers.items() : 
            pipeline = Pipeline([('scale', StandardScaler()), (name, clf)])
            print('\n'+ name + ', 5fold_CV :')
            cv_scores=cross_val_score(pipeline, 
                            X=train_samples,
                            y=train_labels,
                            cv=5,
                            n_jobs=-1,
                            verbose=1)
            dic[name+'_5FCV'] = cv_scores.mean()
            
            #print score per fol & the mean of it
            print('Average accuracy = %.02f \n' % (cv_scores.mean()))
            print('Average Accuracy std = %.02f \n'% (cv_scores.std()))
            print('Accuracy per fold:', cv_scores, sep='\n')
            
            for key, chunk in chunks.items() : 
                print('\n'+ name + ','+ key + ' :' )
                cv_scores = cross_val_score(pipeline,
                            X=train_samples,
                            y=train_labels,
                            groups=chunk,
                            cv=LeaveOneGroupOut(),
                            n_jobs=-1,
                            verbose=1)
                dic[name+'_'+key]= cv_scores.mean()
                
                #print score per fol & the mean of it
                print('Average accuracy = %.02f \n' % (cv_scores.mean()))
                print('Average Accuracy std = %.02f \n'% (cv_scores.std()))
            
def test_baseline(X_train, X_test, y_train, y_test, clf_dic):
    if not sys.warnoptions : 
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARGNINGS"]=('ignore::ConvergenceWarning')
        for name, clf in clf_dic.items():
            std_samples_train =StandardScaler().fit_transform(X_train)
            std_sample_test=StandardScaler().fit_transform(X_test)
            clf.fit(std_samples_train, y_train)
            y_pred= clf.predict(std_sample_test)
            acc=accuracy_score(y_test, y_pred)
            print('Test accuracy Score for ',name, ' : ',  acc, '\n')   
            cm=confusion_matrix(y_test, y_pred)
            disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
            disp.plot()
            plt.xticks(rotation=60)
            plt.show()
        


def tuned_run(pipeline, grid,chunks, train_samples, train_labels, type_, dic, name_clf) :
    
    if not sys.warnoptions : 
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARGNINGS"]=('ignore::ConvergenceWarning')
        print('\n'+ name_clf + ', 5fold CV : \n')
        
        grid_result = GridSearchCV(pipeline, param_grid = grid, scoring = 'accuracy', verbose=1, n_jobs=1, cv = 5)
        grid_result.fit(train_samples, train_labels)
        
        print('Mean test score :' , grid_result.cv_results_['mean_test_score'], '\n')
        print('Test score std  :', grid_result.cv_results_['std_test_score'], '\n')
        print('Best Score: ', grid_result.best_score_) 
        print('Best Params: ', grid_result.best_params_) 
        dic[name_clf+'_5FCV_'+type_]=grid_result.best_score_
        
        for key, chunk in chunks.items():
            print('\n'+name_clf+ ' , ' +  key +' : \n')
            grid_result = GridSearchCV(pipeline, param_grid = grid, scoring = 'accuracy', verbose=1, n_jobs=-1, 
                                   cv =LeaveOneGroupOut())
            grid_result.fit(train_samples, train_labels ,groups = chunk)
            print('Mean test score :' , grid_result.cv_results_['mean_test_score'], '\n')
            print('Test score std  :', grid_result.cv_results_['std_test_score'], '\n')
            
            dic[name_clf+'_'+key+type_]= grid_result.best_score_
            print('Best Score: ', grid_result.best_score_) #Mean cross-validated score of the best_estimator
            print('Best Params: ', grid_result.best_params_) 
        
        
def evaluation_test (pipeline, X, y, cv_, groups=None):
    if not sys.warnoptions : 
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARGNINGS"]=('ignore::ConvergenceWarning')
        null_cv_scores=permutation_test_score(estimator = pipeline, 
                                          X=X,
                                          y=y,
                                          groups=groups,
                                          cv=cv_, 
                                          n_permutations=10,
                                          n_jobs=1,
                                          verbose=1)
        print('Prediction accuracy : %0.2f'% null_cv_scores[0], '\n',
              'p-value: %0.04f'%(null_cv_scores[2]))