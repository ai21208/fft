from fft_main import *

def bench_1d(func, N: int, num_runs: int, cores: list):
    t = np.linspace(0, 100, N)

    # fig, ax = plt.subplots()
    sig = gen_signal(t, [1, 7, 9], [3, 5, 1], 2)

    means = []
    std = []

    df = pd.DataFrame()

    for num_cores in cores:
        nb.set_num_threads(num_cores)
        sig = gen_signal(t, [1, 7, 9], [3, 5, 1], 2)
        runtimes = benchmark("func(sig)", globals=locals(), number=num_runs)
        print(
            f"{num_cores=} ({describe(runtimes)["mean"]} +/- {describe(runtimes)["std"]})s"
        )
        means.append(describe(runtimes)["mean"])
        std.append(describe(runtimes)["std"])
        df[f"{num_cores}"] = runtimes

        cores_str = ",".join(str(core) for core in cores)
        df.to_csv(f"2^{np.log2(N)}-{num_runs}-({cores_str})-iterative.csv")

    # ax.plot(cores, means, marker="o")
    # ax.errorbar(cores, means, yerr=std, fmt='none', capsize=3)


def bench_2d(func, N: int, M: int, num_runs: int, cores):
    cores = list(cores)
    # setup input for the 2D FFT
    x, y = np.meshgrid(np.linspace(-N // 2, N // 2, N), np.linspace(-M // 2, M // 2, M))

    gaussian = scipy.stats.multivariate_normal(
        mean=(0, 0), cov=[[2**5, 0], [0, 2**5]], allow_singular=False
    )
    pos = np.dstack((x, y))
    z = gaussian.pdf(pos).reshape(N, M)

    # dataframe for stats & plotting
    df = pd.DataFrame()

    for i, num_cores in enumerate(cores):
        nb.set_num_threads(num_cores)
        runtimes = benchmark("func(z)", locals(), num_runs)
        print(
            f"{num_cores=} ({describe(runtimes)["mean"]} +/- {describe(runtimes)["std"]})s"
        )

        df[f"{num_cores}"] = runtimes
        cores_str = ",".join(str(core) for core in cores[: i + 1])
        df.to_csv(f"({int(np.log2(N))},{int(np.log2(M))})-{num_runs=}-cores={cores_str}-2d.csv")

    print("2-D Benchmark complete")


def bench_2d_inputsize(func, num_runs, num_cores):
    # dataframe for stats & plotting
    df = pd.DataFrame()
    nb.set_num_threads(num_cores)

    for exponent in range(2,12):
        N = 2**exponent
        M = 2**exponent
        # setup input for the 2D FFT
        x, y = np.meshgrid(np.linspace(-N // 2, N // 2, N), np.linspace(-M // 2, M // 2, M))

        gaussian = scipy.stats.multivariate_normal(
            mean=(0, 0), cov=[[2**5, 0], [0, 2**5]], allow_singular=False
        )
        pos = np.dstack((x, y))
        z = gaussian.pdf(pos).reshape(N, M)

        runtimes = benchmark("func(z)", locals(), num_runs)
        print(
            f"{num_cores=} ({describe(runtimes)["mean"]} +/- {describe(runtimes)["std"]})s"
        )

        df[f"{exponent}"] = runtimes
        df.to_csv(f"{num_runs=}-{num_cores=}-2d.csv")

    print("2-D Input size benchmark complete")

if __name__ == "__main__":
    # bench_2d(fft_nd, 2**22, 2**6, 20, range(8,0,-1))
    bench_2d_inputsize(fft_nd, 100, 8)
