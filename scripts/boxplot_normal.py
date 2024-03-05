import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent.resolve().joinpath("data")
OUTPUT = pathlib.Path(__file__).parent.parent.resolve().joinpath("figs")

DATA_FILE = ROOT.joinpath("data_normal.csv")

if not DATA_FILE.is_file():
    import preprocess

df = pd.read_csv(DATA_FILE)

fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
bp = ax.boxplot(df.iloc[:, :36])
plt.savefig(OUTPUT.joinpath("boxplot_normal.jpg"), bbox_inches = "tight", transparent = True, dpi = 600)
