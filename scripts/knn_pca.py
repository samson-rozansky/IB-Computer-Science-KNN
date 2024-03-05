# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pandas as pd

from tqdm import tqdm

ROOT = pathlib.Path(__file__).parent.parent.resolve().joinpath("data")
OUTPUT = pathlib.Path(__file__).parent.parent.resolve().joinpath("figs\pca")

DATA_FILE = ROOT.joinpath("data_normal.csv")

if not DATA_FILE.is_file():
    import preprocess

data = pd.read_csv(DATA_FILE)

# Create feature and target arrays
X = data.drop(columns = ['Target'])
y = data['Target']

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
			X, y, test_size = 0.2, random_state=42)

pca = PCA(n_components=2)
x = pca.fit_transform(X_train)
y = y_train.to_numpy()
x_test = pca.fit_transform(X_test)


def train_KNN(k):
    model = KNeighborsClassifier(n_neighbors=k, p = 2)
    model = model.fit(x, y)
    predictions = model.predict(x_test)

    plot_decision_regions(x, y.astype(np.int_), clf = model, legend = 2)
    plt.savefig(OUTPUT.joinpath("pca_" + str(k) + ".jpg"), bbox_inches = "tight", transparent = True, dpi = 600)
    plt.clf()

for i in tqdm(range(1, 51)):
    train_KNN(i)