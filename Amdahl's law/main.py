# data from https://github.com/karlrupp/microprocessor-trend-data/blob/master/50yrs/cores.dat

import numpy as np
import matplotlib.pyplot as plt
import pprint

fig, ax = plt.subplots()

def amdahl(t, cores):
    one_over_s = (1 - t) + t/cores
    return 1/one_over_s

cores = [4, 16, 64]


t = np.linspace(0,1,1000)
for core in cores:
    ax.plot(t, amdahl(t, core), label=f"{core} cores")

ax.set(xlim=[0,1], ylabel=r"Speedup $S_{overall}$", xlabel=r"$t_{parallel}$")
ax.set_yscale("log", base=2)
ax.legend(loc="upper left")

fig.set_size_inches(5, 4)

plt.grid()
plt.show()

import mpl_typst.as_default
fig.savefig("Amdahl's law/fig.typ")