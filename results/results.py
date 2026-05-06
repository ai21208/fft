import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def concanenate_results(output_filename, files):
    main_df = pd.DataFrame()
    for file in files:
        file_df = pd.read_csv(file)
        for col in file_df.columns:
            if col != "Unnamed: 0":
                main_df[col] = file_df[col]

    main_df.to_csv(output_filename)


def speedup(df, ax, cores=tuple(range(1, 9))):
    mean_one_core = df["1"].mean()
    std_one_core = df["1"].std()

    cores = np.array(cores)
    means = np.array([df[f"{core}"].mean() for core in cores])
    errors = np.array([df[f"{core}"].std() for core in cores])

    y = mean_one_core / means
    ax.plot(cores, y, marker="o", label="Speedup")
    errors_prop = std_one_core / mean_one_core + errors / means

    print(f"{y=} {errors_prop=}")

    ax.errorbar(cores, y, yerr=errors_prop, fmt="none", capsize=3)


def speedup_plot_1d():
    fig, (ax1, ax2) = plt.subplots(2)

    df_iterative = pd.read_csv("results/2^22-500-iterative.csv")
    speedup(df_iterative, ax1)

    df_recursive = pd.read_csv("results/2^22-100-recursive.csv")
    speedup(df_recursive, ax2)

    ax1.set(ylim=[0, 2.3], xlabel="Cores", ylabel="Relative Speedup")
    ax2.set(ylim=[0, 2], xlabel="Cores", ylabel="Relative Speedup")

    fig.tight_layout()
    plt.show()

def speedup_2d(df, ax, cores):
    mean_one_core = df["1"].mean()
    std_one_core = df["1"].std()

    cores = np.array(cores)
    means = np.array([df[f"{core}"].mean() for core in cores])
    errors = np.array([df[f"{core}"].std() for core in cores])

    y = mean_one_core / means
    ax.plot(cores, y, marker="o", label="Speedup")
    errors_prop = std_one_core / mean_one_core + errors / means

    ax.errorbar(cores, y, yerr=errors_prop, fmt="none", capsize=3)

    # first 12 data points since the residual of the linear regression
    # increases sharply after this point
    fit_cores = cores[:12]
    (m, c), cov = np.polyfit(fit_cores, y[:12], 1, cov=True)
    ax.plot(fit_cores, m*fit_cores + c, label="Linear Regression (1-16 cores)")
    print(f"{m=} {c=} {np.sqrt(cov[0,0])=}")

    ax.legend()

def speedup_plot_2d(filename, cores):
    fig, ax1 = plt.subplots()
    df = pd.read_csv(filename)

    speedup_2d(df, ax1, cores)

    ax1.set(xlabel="Cores", ylabel="Relative Speedup")

    fig.tight_layout()
    plt.show()

    import mpl_typst.as_default
    fig.savefig("BC2d.typ")


def times_plot():
    fig, ax1 = plt.subplots(1)

    df_iterative = pd.read_csv("results/1-D Results/2^22-500-iterative.csv")
    df_recursive = pd.read_csv("results/1-D Results/2^22-100-recursive.csv")

    cores = list(range(1, 9))

    means_iter = np.array([df_iterative[f"{core}"].mean() for core in cores])
    errors_iter = np.array([df_iterative[f"{core}"].std() for core in cores])

    means_recur = np.array([df_recursive[f"{core}"].mean() for core in cores])
    errors_recur = np.array([df_recursive[f"{core}"].std() for core in cores])

    print(f"{means_recur.mean()/means_iter.mean()}")

    ax1.plot(cores, means_recur, label="Recursive", marker="o")
    ax1.errorbar(cores, means_recur, yerr=errors_recur, fmt="none", capsize=3)

    iter_line, *_ = ax1.plot(cores, means_iter, label="Iterative", marker="o")
    ax1.errorbar(
        cores,
        means_iter,
        yerr=errors_iter,
        fmt="none",
        capsize=3,
        ecolor=iter_line.get_color(),
    )

    ax1.set(xlabel="Cores", ylabel="Execution time (s)", ylim=[0, 4.9])
    ax1.legend()

    fig.tight_layout()
    fig.set_size_inches(5, 4)

    import mpl_typst.as_default
    fig.savefig("fig.typ")


def inputsize_fig():
    fig, ax = plt.subplots()

    df = pd.read_csv("results/2D-Input-size/num_runs=100-num_cores=8-2d.csv")

    means = [df[col].mean() for col in df.columns if col != "Unnamed: 0"]
    errors = [df[col].std() for col in df.columns if col != "Unnamed: 0"]

    x = 2 ** np.array(range(2, 12))
    ax.plot(x, means)
    # ax.errorbar(x, means, yerr=errors, fmt="none", capsize=3)
    ax.set_yscale("log")
    ax.set_xscale("log")

    plt.show()


speedup_plot_2d(
    "results/BC-2d/(22,6)-num_runs=100-2d.csv", list(range(1, 9)) + list(range(10, 30, 2))
)
# concanenate_results(
#     "results/BC-2d-(22,6)/test.csv",
#     [
#         f"results/BC-2d-(22,6)/(22,6)-num_runs=100-cores={i}-2d.csv"
#         for i in list(range(1,11)) + list(range(12, 30, 2))
#     ],
# )

# inputsize_fig()
times_plot()