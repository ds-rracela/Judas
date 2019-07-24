import numpy as np
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split
from .utils import progressbar

class TrainSVM():

    C = [1e-8, 1e-4, 1e-3, 1e-2, 0.1, 0.2,0.4, 0.75, 1, 1.5, 3, 5, 10, 15,  20, 100, 300, 1000, 5000]

    def __init__(self, X, y, kernel, Number_trials):
    
        score_train = []
        score_test = []
        weighted_coefs_seeds = []
        self.kernel = kernel
        
        for seed in progressbar(range(Number_trials), 'Computing: ', 'Seed'):
            training_accuracy = []  
            test_accuracy = []
            weighted_coefs = []
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
            for c in self.C:
                svr = SVR(kernel=self.kernel, C=c, gamma='auto').fit(X_train, y_train)
                training_accuracy.append(svr.score(X_train, y_train))
                test_accuracy.append(svr.score(X_test, y_test))
                
                coefs = svr.coef_
                weighted_coefs.append(coefs)
                    
            score_train.append(training_accuracy)
            score_test.append(test_accuracy)
            weighted_coefs_seeds.append(weighted_coefs)

        self.score = np.mean(score_test, axis=0)
        mean_coefs=np.mean(weighted_coefs_seeds, axis=0) #get the mean of the weighted coefficients over all the trials 
        top_weights = np.abs(mean_coefs)[np.argmax(self.score)]
        top_pred_feature_index = np.argmax(top_weights)
        self.top_predictor = X.columns[top_pred_feature_index]        
            
        return

    def result(self):
        return ['SVR ({0})'.format(self.kernel), '{:.2%}'.format(np.amax(self.score)), \
                'C = {0}'.format(self.C[np.argmax(self.score)]), self.top_predictor]


class TrainNSVM():

    C = [1e-8, 1e-4, 1e-3, 1e-2, 0.1, 0.2,0.4, 0.75, 1, 1.5, 3, 5, 10, 15,  20, 100, 300, 1000, 5000]

    def __init__(self, X, y, kernel, Number_trials):
    
        score_train = []
        score_test = []
        self.kernel = kernel
        
        for seed in progressbar(range(Number_trials), 'Computing: ', 'Seed'):
            training_accuracy = []  
            test_accuracy = []
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
            for c in self.C:
                svr = SVR(kernel=self.kernel, C=c, gamma='auto').fit(X_train, y_train)
                training_accuracy.append(svr.score(X_train, y_train))
                test_accuracy.append(svr.score(X_test, y_test))
                
            score_train.append(training_accuracy)
            score_test.append(test_accuracy)

        self.score = np.mean(score_test, axis=0)
        self.top_predictor ='NA'      
            
        return

    def result(self):
        return ['SVR ({0})'.format(self.kernel), '{:.2%}'.format(np.amax(self.score)), \
                'C = {0}'.format(self.C[np.argmax(self.score)]), self.top_predictor]