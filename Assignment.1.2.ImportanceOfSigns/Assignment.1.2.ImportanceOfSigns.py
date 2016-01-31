import pandas
import numpy as np 
from sklearn.tree import DecisionTreeClassifier

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

def convertSex(sex):
  if (sex == 'male'):    
    return 1
  else:    
    return 0

data = data[np.isfinite(data['Age'])]
data['Sex'] = data['Sex'].map({'male': 1, 'female': 0})

signs = data[['Pclass', 'Fare', 'Age', 'Sex']]

print signs

targets = data[['Survived']]
print targets

clf = DecisionTreeClassifier(random_state=241)
clf.fit(signs, targets)

importances = clf.feature_importances_
print importances