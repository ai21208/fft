# data from https://github.com/karlrupp/microprocessor-trend-data/blob/master/50yrs/cores.dat

import numpy as np
import matplotlib.pyplot as plt
import pprint

data = []

for file in ["cores.dat","frequency.dat","transistors.dat"]:
    file_data = []
    with open(f"moores_law/{file}") as f:
        file_data.extend([line.split() for line in f.readlines()])
    data.append(file_data)

pprint.pprint(data)

cores, frequency, transistors = data

cores = np.array(cores, dtype=np.float64).T
frequency = np.array(frequency, dtype=np.float64).T
transistors = np.array(transistors, dtype=np.float64).T

fig, ax = plt.subplots()


ax.set(xlabel="Year", xticks=list(range(1970, 2026, 10)), xlim=[1971, 2023])
ax.scatter(*cores, label="Cores", s=7)
ax.scatter(*frequency, label="Frequency (MHz)", s=7, marker=",")
ax.scatter(*transistors, label="Transistors (thousands)", s=7, marker="^")
ax.set_yscale("log")

ax.legend()

fig.set_size_inches(6, 4.8)

plt.grid()
plt.show()

import mpl_typst.as_default
fig.savefig("moores_law/fig.typ")