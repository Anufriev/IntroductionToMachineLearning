import pandas
import numpy as np 
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor

boston = load_boston()
skaledData = scale(boston.data)

for p in np.linspace(1.0, 10.0, num=200):
    clf = KNeighborsRegressor(metric='minkowski', n_neighbors=5, weights='distance', p=p)
    print clf.fit(skaledData, boston.target)
    print '------------------------------------------'