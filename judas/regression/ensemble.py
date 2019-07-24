import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from .utils import progressbar

class TrainDecisionTree():

    #maxdepth_settings = range(1, 20) # try n_neighbors from 1 to 50

    def __init__(self, X, y, Number_trials, maxdepth_settings=range(1, 20)):
        self.maxdepth_settings = maxdepth_settings
        lahat_training = pd.DataFrame()
        lahat_test = pd.DataFrame()

        for seedN in progressbar(range(1,Number_trials+1,1), 'Computing: ', 'Seed'):
            X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=seedN)

            training_accuracy = []
            test_accuracy = []

            for depth in maxdepth_settings:   
                tree = DecisionTreeRegressor(max_depth=depth, random_state=42)  # build the model
                tree.fit(X_train, y_train)

                training_accuracy.append(tree.score(X_train, y_train)) # record training set accuracy
                test_accuracy.append(tree.score(X_test, y_test))   # record generalization accuracy

            lahat_training[seedN]=training_accuracy
            lahat_test[seedN] = test_accuracy
                
        self.score = np.mean(lahat_test.values, axis=1)      

        # get top predictor
        best_depth = maxdepth_settings[np.argmax(self.score)]
        tree = DecisionTreeRegressor(max_depth=best_depth, random_state=42)  # build the model
        tree.fit(X_train, y_train)
        self.top_predictor = X.columns[np.argmax(tree.feature_importances_)]

        #self.top_predictor='NA'
        return

    def result(self):
        return ['Decision Trees', '{:.2%}'.format(np.amax(self.score)), \
                'depth = {0}'.format(self.maxdepth_settings[np.argmax(self.score)]), self.top_predictor]


class TrainRandomForest():
    # n_estimators_settings = range(1, 20) # try n_neighbors from 1 to 50

    def __init__(self,X,y, Number_trials, n_estimators_settings=range(1,20)):

        self.n_estimators_settings = n_estimators_settings
        lahat_training = pd.DataFrame()
        lahat_test = pd.DataFrame()
        for seedN in progressbar(range(1,Number_trials+1,1), 'Computing: ', 'Seed'):
            X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=seedN)

            training_accuracy = []
            test_accuracy = []
            
            for estimator in n_estimators_settings:   
                forest = RandomForestRegressor(n_estimators=estimator, random_state=0, max_features='auto')
                forest.fit(X_train, y_train)
                training_accuracy.append(forest.score(X_train, y_train)) # record training set accuracy
                test_accuracy.append(forest.score(X_test, y_test))   # record generalization accuracy

            lahat_training[seedN]=training_accuracy
            lahat_test[seedN] = test_accuracy
        
        self.score = np.mean(lahat_test.values, axis=1)

        # get top predictor
        best_estimator = n_estimators_settings[np.argmax(self.score)]
        forest = RandomForestRegressor(n_estimators=best_estimator, random_state=0, max_features='auto')  # build the model
        forest.fit(X_train, y_train)
        self.top_predictor = X.columns[np.argmax(forest.feature_importances_)]
        #self.top_predictor='NA'

    def result(self):
        return ['Random Forest', '{:.2%}'.format(np.amax(self.score)), \
                'n-estimator = {0}'.format(self.n_estimators_settings[np.argmax(self.score)]), self.top_predictor]

class TrainGBM():

    # maxdepth_settings = range(1, 10) # try n_neighbors from 1 to 50

    def __init__(self,X,y, Number_trials, maxdepth_settings=range(1, 10)):

        self.maxdepth_settings=maxdepth_settings
        lahat_training = pd.DataFrame()
        lahat_test = pd.DataFrame()
        for seedN in progressbar(range(1,Number_trials+1,1), 'Computing: ', 'Seed'):
            X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=seedN)

            training_accuracy = []
            test_accuracy = []

            for depth in maxdepth_settings:   

                gbrt = GradientBoostingRegressor(max_depth=depth, learning_rate=0.01, random_state=0)  # build the model
                gbrt.fit(X_train, y_train)

                training_accuracy.append(gbrt.score(X_train, y_train)) # record training set accuracy
                test_accuracy.append(gbrt.score(X_test, y_test))   # record generalization accuracy

            lahat_training[seedN]=training_accuracy
            lahat_test[seedN] = test_accuracy
            
        self.score = np.mean(lahat_test.values, axis=1)

        # get top predictor
        best_depth = maxdepth_settings[np.argmax(self.score)]
        gbrt = GradientBoostingRegressor(max_depth=best_depth, learning_rate=0.01, random_state=0)  # build the model
        gbrt.fit(X_train, y_train)
        self.top_predictor = X.columns[np.argmax(gbrt.feature_importances_)]
        #self.top_predictor='NA'
        return
    
    def result(self):
        return ['Gradient Boosting Method', '{:.2%}'.format(np.amax(self.score)), \
                'depth = {0}'.format(self.maxdepth_settings[np.argmax(self.score)]), self.top_predictor]