# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets, neighbors
from mlxtend.plotting import plot_decision_regions

import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent.resolve().joinpath("data")

data = pd.read_csv(ROOT.joinpath("data.csv"))

# Create feature and target arrays
X = data.drop(columns = ['Target'])
y = data['Target']

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
			X, y, test_size = 0.2, random_state=42)

def knn_comparison(data, k):
    x = data.drop(columns = ['Target']).values
    y = data['Target'].values
    clf = neighbors.KNeighborsClassifier(n_neighbors=k)
    clf.fit(x, y)

    plot_decision_regions(x, y, clf=clf, legend=2)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Knn with K='+ str(k))
    plt.show()

for i in [1, 3, 5, 10]:
    knn_comparison(data, i)
