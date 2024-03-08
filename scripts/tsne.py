import matplotlib.pyplot as plt
from matplotlib import colormaps
import pathlib
import pandas as pd
import plotly.express as px

from sklearn import metrics, manifold
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE

ROOT = pathlib.Path(__file__).parent.parent.resolve().joinpath("data")
OUTPUT = pathlib.Path(__file__).parent.parent.resolve().joinpath("figs")

DATA_FILE = ROOT.joinpath("data_normal.csv")

if not DATA_FILE.is_file():
    import preprocess

data = pd.read_csv(DATA_FILE)

# Create feature and target arrays
X = data.drop(columns = ['Target'])
y = data['Target']

tsne = TSNE(n_components=2, random_state=42, init='pca', perplexity=10, early_exaggeration=24, metric='l1')
X_tsne = tsne.fit_transform(X)

x_0 = []
y_0 = []
x_1 = []
y_1 = []

for i in range(y.size):
    if (y[i] == 0):
        x_0.append(X_tsne[i, 0])
        y_0.append(X_tsne[i, 1])
    else:
        x_1.append(X_tsne[i, 0])
        y_1.append(X_tsne[i, 1])

plt.scatter(x=x_0, y=y_0, c='tab:red', label='Dropout')
plt.scatter(x=x_1, y=y_1, c='tab:green', label='Graduate')
plt.ylabel('Second t-SNE')
plt.xlabel('First t-SNE')
plt.legend()
plt.savefig(OUTPUT.joinpath("tsne.jpg"), bbox_inches = "tight", transparent = True, dpi = 600)

