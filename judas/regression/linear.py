import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from .utils import progressbar

class TrainLinear():
    
    alpha = [1e-8, 1e-4, 1e-3, 1e-2, 0.1, 0.2,0.4, 0.75, 1, 1.5, 3, 5, 10, 15,  20, 100, 300, 1000, 5000]

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

            if reg == 'linear':
                lr = LinearRegression().fit(X_train, y_train)
                training_accuracy = [lr.score(X_train, y_train)]
                test_accuracy = [lr.score(X_test, y_test)]
                weighted_coefs = [lr.coef_[0]]

            elif reg in ['lasso', 'ridge']:
                for alpha_run in self.alpha:
                    # if reg == 'linear':
                    #     lr = LinearRegression().fit(X_train, y_train)
                    if reg == 'lasso':
                        lr = Lasso(alpha=alpha_run).fit(X_train, y_train)
                    elif reg == 'ridge':
                        lr = Ridge(alpha=alpha_run).fit(X_train, y_train)

                    training_accuracy.append(lr.score(X_train, y_train))
                    test_accuracy.append(lr.score(X_test, y_test))
            
                    coefs = lr.coef_[0] 
                    weighted_coefs.append(coefs) #append all the computed coefficients per trial
                    
            score_train.append(training_accuracy)
            score_test.append(test_accuracy)
            weighted_coefs_seeds.append(weighted_coefs)

        self.score = np.mean(score_test, axis=0)

        mean_coefs=np.mean(weighted_coefs_seeds, axis=0) #get the mean of the weighted coefficients over all the trials 
        top_weights = np.abs(mean_coefs)[np.argmax(self.score)]
        top_pred_feature_index = np.argmax(top_weights)
        self.top_predictor = X.columns[top_pred_feature_index]  
            
        return

    # def result(self):
    #     return ['Logistic ({0})'.format(self.reg), np.amax(self.score), \
    #             'C = {0}'.format(self.C[np.argmax(self.score)]), self.top_predictor]

    def result(self):
        if self.reg != 'linear':
            return ['{}'.format(self.reg.title()), '{:.2%}'.format(np.amax(self.score)), \
                'alpha = {0}'.format(self.alpha[np.argmax(self.score)]), self.top_predictor]
        return ['{}'.format(self.reg.title()), '{:.2%}'.format(np.amax(self.score)), \
                'NA', self.top_predictor]
