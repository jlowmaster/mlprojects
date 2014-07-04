#! /usr/bin/env python

import numpy as np
import csv
import sklearn
import random
from sklearn import preprocessing
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn import cross_validation


def main():

    fn = './data/letter-recognition.data'
    dat = np.loadtxt(fn, delimiter=',', dtype='str')

    X = dat[:, 1:]
    Y = dat[:, 0]
    
    X_train, X_val, y_train, y_val = cross_validation.train_test_split(
    X, Y, test_size=0.25, random_state=0)
    
    X_train.dtype = np.int16
    X_val.dtype = np.int16

    print
    print
    print "Random Forest CVs"
    print
    print

    scores = ['precision', 'recall']
    tuned_parameters = [{'n_estimators': [35, 45, 55],
                         'max_features': [4,5],
                         'min_samples_split': [4, 5]
                         }]
    
    for score in scores:
        clf = GridSearchCV(RandomForestClassifier(min_samples_leaf=2), 
                           tuned_parameters, cv=5, scoring=score)
        clf.fit(X_train, y_train)
        y_true, y_pred = y_val, clf.predict(X_val)
        print clf.best_estimator_
        print metrics.classification_report(y_true, y_pred)
    
    print
    print
    print "KNN CVs"
    print
    print 
    
    scores = ['precision', 'recall']
    tuned_parameters = [{'n_neighbors': [3,4,5],
                         'leaf_size': [20, 30, 40]
                         }]
    for score in scores:
        clf = GridSearchCV(KNeighborsClassifier(weights='distance', 
                                                algorithm='kd_tree'), tuned_parameters, 
                           cv=5, scoring=score)
        clf.fit(X_train, y_train)
        y_true, y_pred = y_val, clf.predict(X_val)
        print clf.best_estimator_
        print metrics.classification_report(y_true, y_pred)

if __name__ == '__main__':
    main()
