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
OUTPUT = pathlib.Path(__file__).parent.parent.resolve().joinpath("figs")

DATA_FILE = ROOT.joinpath("data_normal.csv")

if not DATA_FILE.is_file():
    import preprocess

data = pd.read_csv(DATA_FILE)

# Create feature and target arrays
X = data.drop(columns = ['Target'])
y = data['Target']

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print('Evaluation Metrics:')
print('Accuracy: ' + str(metrics.accuracy_score(y_test, predictions)))
print('Recall: ' + str(metrics.recall_score(y_test, predictions)))
print('F1 Score: ' + str(metrics.f1_score(y_test, predictions)))
print('Precision: ' + str(metrics.precision_score(y_test, predictions)))
    
# Print Confusion Matrix
print('Confusion Matrix:')
sns.set(font_scale = 2)
cm = confusion_matrix(y_test, predictions, labels = None)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = None)
disp.plot()
plt.grid(False)
plt.show()