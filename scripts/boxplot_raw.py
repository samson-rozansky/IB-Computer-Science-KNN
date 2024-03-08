import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent.resolve().joinpath("data")
OUTPUT = pathlib.Path(__file__).parent.parent.resolve().joinpath("figs")

data = pd.read_csv(ROOT.joinpath("data_raw.csv"))

fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
bp = ax.boxplot(data.iloc[:, :36])
plt.show()
#plt.savefig(OUTPUT.joinpath("boxplot_raw.jpg"), bbox_inches = "tight", transparent = True, dpi = 600)
