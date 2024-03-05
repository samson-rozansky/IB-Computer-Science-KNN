import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib

ROOT = pathlib.Path(__file__).parent.parent.resolve().joinpath("data")
OUTPUT = pathlib.Path(__file__).parent.parent.resolve().joinpath("figs")

DATA_FILE = ROOT.joinpath("data_raw.csv")

df = pd.read_csv(DATA_FILE, delimiter = ";")

correlations = np.round(df.corr().to_numpy(), decimals=2)

#plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')
#plt.style.use(['dark_background'])

plt.rcParams.update({'font.size': 5})

fig, ax = plt.subplots()
im = ax.imshow(correlations, cmap = 'inferno')

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(list(df.columns))), labels=list(df.columns))
ax.set_yticks(np.arange(len(list(df.columns))), labels=list(df.columns))

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
"""
for i in range(len(list(df.columns))):
    for j in range(len(list(df.columns))):
        text = ax.text(j, i, correlations[i, j], ha="center", va="center", color="aqua")
"""

fig.tight_layout()

plt.savefig(OUTPUT.joinpath("heatmap.jpg"), bbox_inches = "tight", transparent = True, dpi = 600)
