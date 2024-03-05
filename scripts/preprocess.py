import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent.resolve().joinpath("data")

data = pd.read_csv(ROOT.joinpath("data_raw.csv"), delimiter = ';')

data_normal = data.copy()
for column in data_normal.columns[:36]:
    data_normal[column] = (data_normal[column] - data_normal[column].mean())/data_normal[column].std(ddof=0)

ret = []
for index, row in data_normal.iterrows():
    if (row["Target"] != "Enrolled"):
        if (row["Target"] == "Graduate"):
            row["Target"] = 1
        else:
            row["Target"] = 0
        ret.append(row)

df = pd.DataFrame(ret)
df.reset_index()
df.to_csv(ROOT.joinpath("data_normal.csv"), index = False)
