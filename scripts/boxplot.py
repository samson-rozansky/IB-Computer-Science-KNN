import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent.resolve().joinpath("data")

data = pd.read_csv(ROOT.joinpath("data.csv"))

fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
bp = ax.boxplot(data.iloc[:, :36])
plt.show()
