from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent.resolve().joinpath("data")
OUTPUT = pathlib.Path(__file__).parent.parent.resolve().joinpath("figs")

DATA_FILE = ROOT.joinpath("data_normal.csv")

if not DATA_FILE.is_file():
    import preprocess

data = pd.read_csv(DATA_FILE)

# Create feature and target arrays
BruhEmoji = "Marital status,Father's occupation,Educational special needs,Age at enrollment,Curricular units 1st sem (enrolled),Curricular units 2nd sem (without evaluations),Application mode,Application order,Course,Daytime/evening attendance	,Previous qualification,Previous qualification (grade),Nacionality,Mother's qualification,Father's qualification,Mother's occupation,Admission grade,Displaced,Debtor,Tuition fees up to date,Gender,Scholarship holder,International,Curricular units 1st sem (credited),Curricular units 1st sem (evaluations),Curricular units 1st sem (approved),Curricular units 1st sem (grade),Curricular units 1st sem (without evaluations),Curricular units 2nd sem (credited),Curricular units 2nd sem (enrolled),Curricular units 2nd sem (evaluations),Curricular units 2nd sem (approved),Curricular units 2nd sem (grade),Unemployment rate,Inflation rate,GDP,Target"
toDrop = BruhEmoji.split(",")
MAXIMUM = 0
dropped = ""
neighneigh = 0
counter = 1
for i in range(len(toDrop)):
    for j in range(len(toDrop)):
        if(i>=j):
            continue
        for k in range(len(toDrop)):
            if(j>=k):
                continue
            X = data.drop(columns = ['Target',toDrop[i],toDrop[j],toDrop[k]])
            y = data['Target']

            # Split into training and test set
            X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size = 0.2, random_state=42)

            neighbors = np.arange(4, 8)
            train_accuracy = np.empty(len(neighbors))
            test_accuracy = np.empty(len(neighbors))
            print(counter)
            counter += 1
            # Loop over K values
            for i, k in enumerate(neighbors):
                knn = KNeighborsClassifier(n_neighbors=k,n_jobs=1)
                knn.fit(X_train, y_train)
                
                # Compute training and test data accuracy
                train_accuracy[i] = knn.score(X_train, y_train)
                test_accuracy[i] = knn.score(X_test, y_test)
                if(test_accuracy[i]>MAXIMUM):
                    neighneigh = k
                    dropped = toDrop[i] + " " + toDrop[j] + toDrop[k]
                    MAXIMUM = test_accuracy[i]
            print(counter)
            counter += 1
    print(neighneigh,dropped,MAXIMUM) 
            
        
        
print(neighneigh,dropped,MAXIMUM)      
# Generate plot
plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')

plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.savefig(OUTPUT.joinpath("RFE.png"), bbox_inches="tight", transparent = True, dpi = 600)
