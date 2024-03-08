from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import os
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pandas as pd
import seaborn as sns

from sklearn import metrics
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

ROOT = pathlib.Path(__file__).parent.parent.resolve().joinpath("data")
OUTPUT = pathlib.Path(__file__).parent.parent.resolve().joinpath("figs\confusion")

DATA_FILE = ROOT.joinpath("data_normal.csv")

if not DATA_FILE.is_file():
    import preprocess

data = pd.read_csv(DATA_FILE)

# Create feature and target arrays
X = data.drop(columns = ['Target'])
y = data['Target']

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

def test_model(k, cur_metric, X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=k, metric = cur_metric)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    write_file = OUTPUT.joinpath(cur_metric + "_result.txt")
    if not write_file.is_file():
        open(write_file, 'x')
    with open(write_file, 'w') as f:
        f.write('Evaluation Metrics for ' + cur_metric + ':\n')
        f.write('Accuracy: ' + str(metrics.accuracy_score(y_test, predictions)) + '\n')
        f.write('Recall: ' + str(metrics.recall_score(y_test, predictions)) + '\n')
        f.write('F1 Score: ' + str(metrics.f1_score(y_test, predictions)) + '\n')
        f.write('Precision: ' + str(metrics.precision_score(y_test, predictions)) + '\n')
        
    # Print Confusion Matrix
    sns.set(font_scale = 2)
    cm = confusion_matrix(y_test, predictions, labels = None)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = None)
    disp.plot()
    plt.grid(False)
    plt.savefig(OUTPUT.joinpath(cur_metric + "_confusion_matrix.jpg"), bbox_inches = "tight", transparent = True, dpi = 600)
    plt.close()
    return metrics.accuracy_score(y_test, predictions)

metrics_list = ['cosine', 'l1', 'l2']
for m in metrics_list:
    best_acc = 0
    best_k = 0
    for k in np.arange(1, 21):
        cur_acc = test_model(k, m, X_train, X_test, y_train, y_test)
        if (cur_acc > best_acc):
            best_acc = cur_acc
            best_k = k
    print('Best k for ' + m + ': ' + str(best_k))
    test_model(best_k, m, X_train, X_test, y_train, y_test)
