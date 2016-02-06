import pandas
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn import preprocessing

data_frame = pandas.read_csv('wine.data')

y = data_frame['1']
X = data_frame.drop('1', axis=1)
normalizedX = preprocessing.scale(X=X)

kf = KFold(n=len(y), n_folds=5, shuffle=True, random_state=42)

for index in range(50 - 1):
  n = index + 1
  clf = KNeighborsClassifier(n_neighbors=n)
  accurancy1 = cross_validation.cross_val_score(estimator=clf, X=X, y=y, cv=kf, scoring='accuracy')
  accurancy2 = cross_validation.cross_val_score(estimator=clf, X=normalizedX, y=y, cv=kf, scoring='accuracy')
  print n, ' -> ', round(np.average(accurancy1), 2), round(np.average(accurancy2), 2)