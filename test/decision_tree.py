from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus

X = [['A', 1], ['B', 1]]

Y = [0, 1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)



