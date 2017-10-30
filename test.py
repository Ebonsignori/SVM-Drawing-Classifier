from sklearn import svm, datasets
import numpy as np

# import some data to play with
iris = datasets.load_iris()
# Take the first two features. We could avoid this by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target

test = np.array([[3, 7, 5, 5],
         [0, 1, 5, 9],
         [3, 0, 5, 0]])
