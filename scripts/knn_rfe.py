from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pandas as pd
import seaborn as sns

from sklearn import metrics
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

ROOT = pathlib.Path(__file__).parent.parent.resolve().joinpath("data")
OUTPUT = pathlib.Path(__file__).parent.parent.resolve().joinpath("figs\\rfe")

DATA_FILE = ROOT.joinpath("data_normal.csv")

if not DATA_FILE.is_file():
    import preprocess

data = pd.read_csv(DATA_FILE)

features = []
accuracies = []

def RFE(df, m, k):
    best_acc = 0
    feature = 'temp'
    for column in df:
        if (column == 'Target'):
            continue
        temp = df.copy()
        temp.drop(columns = [column], inplace = True)
        X = temp.drop(columns = ['Target'])
        y = temp['Target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
        model = KNeighborsClassifier(n_neighbors=k, metric=m)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cur_acc = metrics.accuracy_score(y_test, y_pred)
        if (cur_acc > best_acc):
            feature = column
            best_acc = cur_acc
    
    df.drop(columns = [feature], inplace = True)
    features.append(feature)
    accuracies.append(best_acc)

    if (len(df.columns) > 2):
        print("Number eliminated: " + str(len(features)))
        print("Number remaining: " + str(len(df.columns)) + "\n")
        RFE(df.copy(), m, k)
        return
    else:
        return

metrics_list = ['cosine', 'l1', 'l2']
k = [16, 10, 6]

for i in range(3):
    RFE(data, metrics_list[i], k[i])
    write_file = OUTPUT.joinpath(metrics_list[i] + "_RFE.txt")
    open(write_file, 'w').close()

    for j in range(len(features)):
        if not write_file.is_file():
            open(write_file, 'x')
        with open(write_file, 'a') as f:
            f.write("Removed: " + features[j] + "\nAccuracy: " + str(accuracies[j]) + "\n\n")
    
    plt.plot(np.arange(len(features), 0, -1), accuracies)
    plt.xlabel('Number of features')
    plt.ylabel('Accuracy')
    
    plt.savefig(OUTPUT.joinpath(metrics_list[i] + "_RFE.jpg"), bbox_inches = "tight", transparent = True, dpi = 600)
    plt.close()

    features = []
    accuracies = []
