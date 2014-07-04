#! /usr/bin/env python

import numpy as np
import csv
import sklearn
import random
from sklearn import preprocessing
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV


## FUNCTIONS ##

def load_data():

    train_file = "./data/train.csv"
    train = np.loadtxt(train_file, delimiter=',', skiprows=1)
    return train

def get_train(train, obs=1000):

    small_train = np.asarray(random.sample(train, obs))
    y_train = small_train[:,0]
    X_train = small_train[:, 1:]
    return X_train, y_train

def get_val(train, obs=10000):

    val_set = np.asarray(random.sample(train, obs))
    y_val = val_set[:,0]
    X_val = val_set[:, 1:]
    return X_val, y_val

## MAIN ##

def main():

    train = load_data()
    X_train, y_train = get_train(train, obs=10000)
    X_val, y_val = get_val(train, obs=10000)

    scores = ['precision', 'recall']
    tuned_parameters = [{'n_neighbors': [3,4,5],
                         'leaf_size': [20, 30, 40]
                         }]
    for score in scores:
        clf = GridSearchCV(KNeighborsClassifier(weights='distance', algorithm='kd_tree'), tuned_parameters, cv=5, scoring=score)
        clf.fit(X_train, y_train)
        y_true, y_pred = y_val, clf.predict(X_val)
        print clf.best_estimator_
        print metrics.classification_report(y_true, y_pred)
        print metrics.confusion_matrix(y_true, y_pred)

if __name__ == '__main__':
    main()
