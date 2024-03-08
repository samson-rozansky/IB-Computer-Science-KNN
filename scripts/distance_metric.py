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
OUTPUT = pathlib.Path(__file__).parent.parent.resolve().joinpath("figs\distance")

DATA_FILE = ROOT.joinpath("data_normal.csv")

if not DATA_FILE.is_file():
    import preprocess

data = pd.read_csv(DATA_FILE)

# Create feature and target arrays
X = data.drop(columns = ['Target'])
y = data['Target']

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

metrics_list = ['cosine', 'l1', 'l2']
name = ['Cosine', 'L1', 'L2']

for m in range(3):
    neighbors = np.arange(1, 21)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))
    best_acc = 0
    best_k = 0
    for i, k in enumerate(neighbors):
        model = KNeighborsClassifier(n_neighbors=k, metric=metrics_list[m])
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        cur_acc = metrics.accuracy_score(y_test, predictions)
        if (cur_acc > best_acc):
            best_acc = cur_acc
            best_k = k
    
        train_accuracy[i] = model.score(X_train, y_train)
        test_accuracy[i] = model.score(X_test, y_test)    

    print('Best k for ' + name[m] + ': ' + str(best_k))

    model = KNeighborsClassifier(n_neighbors=best_k, metric=metrics_list[m])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    write_file = OUTPUT.joinpath(name[m] + "_result.txt")
    if not write_file.is_file():
        open(write_file, 'x')
    with open(write_file, 'w') as f:
        f.write('Evaluation Metrics for ' + name[m] + ':\n')
        f.write('Accuracy: ' + str(metrics.accuracy_score(y_test, predictions)) + '\n')
        f.write('Recall: ' + str(metrics.recall_score(y_test, predictions)) + '\n')
        f.write('F1 Score: ' + str(metrics.f1_score(y_test, predictions)) + '\n')
        f.write('Precision: ' + str(metrics.precision_score(y_test, predictions)) + '\n')
        
    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions, labels = None)
    #sns.set(font_scale=2)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = None)
    disp.plot()
    plt.grid(False)
    plt.title("KNN using " + name[m] + " metric (k=" + str(best_k) + ")")
    #plt.savefig(OUTPUT.joinpath(name[m] + "_confusion_matrix.jpg"), bbox_inches = "tight", transparent = True, dpi = 600)
    plt.close()

    # graph of k vs accuracy
    plt.plot(neighbors, test_accuracy, label = name[m] + ' Testing dataset Accuracy')
    plt.plot(neighbors, train_accuracy, label = name[m] + ' Training dataset Accuracy')

    plt.legend()
    plt.xlabel('n_neighbors')
    plt.ylabel('Accuracy')
    #plt.savefig(OUTPUT.joinpath(m + "_neighbors_vs_accuracy.jpg"), bbox_inches = "tight", transparent = True, dpi = 600)
    #plt.close()
plt.savefig(OUTPUT.joinpath("neighbors_vs_accuracy.jpg"), bbox_inches = "tight", transparent = True, dpi = 600)