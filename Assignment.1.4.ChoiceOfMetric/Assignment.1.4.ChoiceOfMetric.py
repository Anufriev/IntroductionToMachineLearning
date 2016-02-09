import pandas
import numpy as np 
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import KFold
from sklearn import cross_validation

boston = load_boston()

y = boston.target
X = scale(X=boston.data)

kf = KFold(n=len(y), n_folds=5, shuffle=True, random_state=42)

for p in np.linspace(1.0, 10.0, num=200):
  clf = KNeighborsRegressor(metric='minkowski', n_neighbors=5, weights='distance', p=p)
  accurancy = cross_validation.cross_val_score(estimator=clf, X=X, y=y, cv=kf, scoring='mean_squared_error')
  print p, ' -> ', round(np.average(accurancy), 2)