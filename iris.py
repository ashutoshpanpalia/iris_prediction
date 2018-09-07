import pandas as pd
import numpy as np
from sklearn import model_selection

df= pd.read_csv("C:\Users\Ashutosh\Desktop\Python_cv\Kaggle\iris.data")
print(df.head(10))
print(df.shape)
# Split-out validation dataset
array = df.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.30 #70:30 data split for Training:testing
seed = 25 #random number
X_train, X_validation,Y_train, Y_validation =model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed) #Data splitting 

from sklearn.metrics import accuracy_score #To calculate accuracy

# SVM algorithm for predicton
from sklearn import svm
clf=svm.SVC(kernel='poly')
clf.fit(X_train,Y_train) 
pred=clf.predict(X_validation)
acc=accuracy_score(pred,Y_validation)
print("SVM Accuracy = " ,acc)

# K-Nearest algorithm Neighnors for prediction
from sklearn import neighbors
clf2=neighbors.KNeighborsClassifier(n_neighbors=3)
clf2.fit(X_train,Y_train)
pred2=clf2.predict(X_validation)
acc2=accuracy_score(pred2,Y_validation)
print("KNN accuracy = ", acc2)

# Decision Tree algorithm for prediction
from sklearn import tree
clf3=tree.DecisionTreeClassifier()
clf3.fit(X_train,Y_train)
pred3=clf3.predict(X_validation)
acc3=accuracy_score(pred3,Y_validation)
print("Decision Tree accuracy = ", acc3)

# Naive Bayes algorithm for prediction
from sklearn.naive_bayes import GaussianNB
clf4=GaussianNB()
clf4.fit(X_train,Y_train)
pred4=clf4.predict(X_validation)
acc4=accuracy_score(pred4,Y_validation)
print("Naive Bayes accuracy = ", acc4)

#AdaBoost algorith for prediction
from sklearn.ensemble import AdaBoostClassifier
clf5=AdaBoostClassifier(n_estimators=65,learning_rate=1)
clf5.fit(X_train,Y_train)
pred5=clf5.predict(X_validation)
acc5=accuracy_score(pred5,Y_validation)
print("AdaBoost accuracy = ", acc5)



