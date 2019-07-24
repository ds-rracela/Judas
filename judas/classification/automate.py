import time
from datetime import timedelta
import pandas as pd
import numpy as np
import warnings

from .knn import TrainKNN
from .logistic import TrainLogistic
from .svm import TrainSVM, TrainNSVM, TrainNSVMPoly
from .ensemble import TrainDecisionTree, TrainRandomForest, TrainGBM

class Judas():
    results = []
    models = []

    def __init__(self, DEBUG=False):
        self.DEBUG=DEBUG
        return

    def automate(self, X, y, models):
        warnings.filterwarnings("ignore")
        self.models = []
        for model in models:
            # start_time = time.time()
            if model[0] == 'knn':
                print('{}, n neighbors={}'.format(model[0], model[1]))                
                m = TrainKNN(X,y,model[1], model[2])
            elif model[0] == 'logistic':
                print('{}, reg={}'.format(model[0], model[1]))                
                m = TrainLogistic(X,y, reg=model[1], Number_trials=model[2])
            elif model[0] == 'svm':
                print('{}, reg={}'.format(model[0], model[1]))                
                m = TrainSVM(X,y, reg=model[1], Number_trials=model[2])
            elif model[0] == 'nsvm-rbf':
                print('{}'.format(model[0]))                
                m = TrainNSVM(X,y, Number_trials=model[1])
            elif model[0] == 'nsvm-poly':
                print('{}'.format(model[0]))                
                m = TrainNSVMPoly(X,y, Number_trials=model[1])
            elif model[0] == 'ensemble-decisiontree':
                print('{}, max depth={}'.format(model[0], model[2]))                
                m = TrainDecisionTree(X,y,Number_trials=model[1], maxdepth_settings=model[2])
            elif model[0] == 'ensemble-randomforest':
                print('{}, n estimators={}'.format(model[0], model[2]))                
                m = TrainRandomForest(X,y,Number_trials=model[1], n_estimators_settings=model[2])
            elif model[0] == 'ensemble-gbm':
                print('{}, max depth={}'.format(model[0], model[2]))                
                m = TrainGBM(X,y,Number_trials=model[1], maxdepth_settings=model[2])
            else:
                continue
            # elapsed_time_secs = time.time() - start_time
            # print('{} execution time: {}'.format(model[0], timedelta(seconds=round(elapsed_time_secs))))
            self.models.append(m)

    def score(self):
        cols = ['Machine Learning Method', 'Test Accuracy', 'Best Parameter', 'Top Predictor Variable']
        df = pd.DataFrame(columns=cols)
        #print(self.DEBUG)
        for idx, m in enumerate(self.models):
            if self.DEBUG == True:
                print(idx)
            df.loc[idx] = m.result()
        return df