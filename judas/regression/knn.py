import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from .utils import progressbar

class TrainKNN():

    # neighbors_settings = range(1,70)

    def __init__(self, X, y, Number_trials, neighbors_settings=range(1,70)):
        score_train = []
        score_test = []
        self.neighbors_settings = neighbors_settings

        for seed in progressbar(range(Number_trials), 'Computing: ', 'Seed'):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
            
            acc_train = []
            acc_test = []

            for n_neighbors in self.neighbors_settings:   
                clf = KNeighborsRegressor(n_neighbors=n_neighbors) # build the model 
                clf.fit(X_train, y_train)    
                acc_train.append(clf.score(X_train, y_train))
                acc_test.append(clf.score(X_test, y_test))

            score_train.append(acc_train)
            score_test.append(acc_test)   
            
        self.score = np.mean(score_test, axis=0)

        return

    def result(self):
        return ['kNN', '{:.2%}'.format(np.amax(self.score)), 
                'N_Neighbor = {0}'.format(np.argmax(self.score)+1), 'NA']
