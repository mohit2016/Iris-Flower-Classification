# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 23:21:52 2019

@author: Mohit Uniyal
"""


#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Importing dataset
dataset = pd.read_csv("iris.csv", header=None, names =["sepal_length"	,"sepal_width"	,"petal_length"	,"petal_width",	"species"])
X = dataset.iloc[:,0:4].values
y = dataset.iloc[:,4].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Visualising Dataset
sns.pairplot(dataset)
sns.distplot(dataset['petal_length'])

correlation_matrix= dataset.corr()
sns.heatmap(correlation_matrix)


#Scaling the dataset
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#Spliting the dataset
from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(X, y, test_size =0.3, random_state=0)

#Fitting the Kernal SVM classifier to the model
from sklearn.svm import SVC
classifier = SVC(kernel= "rbf", random_state =0, C=1)
classifier.fit(X_train,y_train)


#Predicting the data
y_pred = classifier.predict(X_test) 


#creating confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#accuracy score of the model
from sklearn import metrics
print(metrics.accuracy_score(y_pred, y_test))

print("Training set score : ",classifier.score(X_train,y_train))
print("Test set score : ",classifier.score(X_test, y_test))

#K-fold cross validation for mean accuracy of 10 folds
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X = X_train, y = y_train, cv=10)
print(accuracies.mean())
#variance of model
print(accuracies.std())

#F1 score of model
from sklearn.metrics import f1_score
print(f1_score(y_test, y_pred, average="micro"))
