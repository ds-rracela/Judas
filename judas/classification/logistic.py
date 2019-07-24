import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from .utils import progressbar

class TrainLogistic():
    
    C = [1e-8, 1e-4, 1e-3, 1e-2, 0.1, 0.2,0.4, 0.75, 1, 1.5, 3, 5, 10, 15,  20, 100, 300, 1000, 5000]

    def __init__(self, X, y, reg, Number_trials):

        score_train = []
        score_test = []
        weighted_coefs_seeds = []
        self.reg = reg
        
        for seed in progressbar(range(Number_trials), 'Computing: ', 'Seed'):
            training_accuracy = []  
            test_accuracy = []
            weighted_coefs=[]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
            for alpha_run in self.C:
                lr = LogisticRegression(C=alpha_run, penalty=reg).fit(X_train, y_train)
                training_accuracy.append(lr.score(X_train, y_train))
                test_accuracy.append(lr.score(X_test, y_test))
                
                coefs = lr.coef_[0] 
                weighted_coefs.append(coefs) #append all the computed coefficients per trial
                    
            score_train.append(training_accuracy)
            score_test.append(test_accuracy)
            weighted_coefs_seeds.append(weighted_coefs)

        self.score = np.mean(score_test, axis=0)

        # print(np.array(weighted_coefs_seeds).shape)        
        mean_coefs=np.mean(weighted_coefs_seeds, axis=0) #get the mean of the weighted coefficients over all the trials 
        # print(mean_coefs.shape)
        # print(self.score.shape)
        # self.top_predictor=X.columns[np.argmax(np.abs(mean_coefs))]
        top_weights = np.abs(mean_coefs)[np.argmax(self.score)]
        # print(top_weights)
        top_pred_feature_index = np.argmax(top_weights)
        # print(top_pred_feature_index)
        self.top_predictor = X.columns[top_pred_feature_index]        
            
        return

    def result(self):
        return ['Logistic ({0})'.format(self.reg), '{:.2%}'.format(np.amax(self.score)), \
                'C = {0}'.format(self.C[np.argmax(self.score)]), self.top_predictor]
