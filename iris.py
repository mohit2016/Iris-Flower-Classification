# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 23:21:52 2019

@author: Mohit Uniyal
"""


#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing dataset
dataset = pd.read_csv("iris.csv")
X = dataset.iloc[:,0:4].values
y = dataset.iloc[:,4].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


#Scaling the dataset
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#Spliting the dataset
from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(X, y, test_size =0.3, random_state=0)

#Fitting the Kernal SVM classifier to the model
from sklearn.svm import SVC
classifier = SVC(kernel= "rbf", random_state =0)
classifier.fit(X_train,y_train)


#Predicting the data
y_pred = classifier.predict(X_test) 

#creating confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#accuracy score of the model
from sklearn import metrics
metrics.accuracy_score(y_pred, y_test)

#K-fold cross validation for mean accuracy of 10 folds
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X = X_train, y = y_train, cv=10)
print(accuracies.mean())
#variance of model
print(accuracies.std())

#F1 score of model
from sklearn.metrics import f1_score
print(f1_score(y_test, y_pred, average="micro"))
