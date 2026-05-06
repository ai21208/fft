import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import time
import timeit
import functools
import pprint
import numba as nb
import scipy.stats


def timer(func):
    start = time.time()
    return_val = func()
    print(f"Function took {time.time() - start} seconds")
    return return_val

def ab_test(other_func):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            out = func(*args, **kwargs)
            agrees = np.allclose(out, other_func(*args, **kwargs))
            print(f"{func.__name__} agrees with {other_func.__name__}: {agrees}")
            return out
        return wrapper
    return decorator

def fft_1d(x):
    """
    Wrapper around _fft_1d function to check the input is valid when first called.
    The if statement doesn't need to be evaluated for every recurive function call.
    """
    if not is_power_of_2(len(x)):
        Exception("Input x must be a power of 2")
    else:
        return _fft_1d(x)

@nb.jit(nopython=True, fastmath=True, nogil=True, cache=True)
def _fft_1d(x: np.ndarray) -> np.ndarray:
    N: int = x.size

    if N == 1:
        return x.astype(np.dtype("complex128"))
    else:
        X_even = _fft_1d(x[::2])  # even elements recursively passed to fft
        X_odd = _fft_1d(x[1::2])  # odd elements recursively passed to fft

        n = np.arange(N, dtype=np.int64)
        exponentials = np.exp(-2j * np.pi * n / N).astype(np.complex128)

        # numba doesn't support np.concatenate
        X = X_even + exponentials[: int(N / 2)] * X_odd
        X = np.append(X, X_even + exponentials[int(N / 2) :] * X_odd)

        return X


def is_power_of_2(n):
    return (n & (n - 1) == 0) and n != 0


def gen_signal(
    t: np.ndarray, frequencies: np.ndarray, amplitudes: np.ndarray, noise: float
):
    N = len(t)
    clean_signal = sum(
        amplitude * np.sin(2 * np.pi * frequency * t)
        for amplitude, frequency in zip(amplitudes, frequencies)
    )
    noisy_signal = clean_signal + np.random.normal(0, noise, N)

    return noisy_signal


def desribe(array: np.ndarray) -> dict:
    summary = {
        "count": len(array),
        "mean": np.mean(array),
        "std": np.std(array),
        "min": np.min(array),
        "25%": np.percentile(array, 25),
        "50%": np.median(array),
        "75%": np.percentile(array, 75),
        "max": np.max(array),
    }
    return summary


def benchmark(code: str, globals, number: int):
    # setup: allow function to complile first time
    timeit.timeit(code, globals=globals, number=1)

    times = np.array(
        [timeit.timeit(code, globals=globals, number=1) for _ in range(number)]
    )
    pprint.pprint(desribe(times))
    return times

def fft_freq(sampling_rate: int, N: int) -> np.ndarray:
    freq = np.arange(N) * sampling_rate / N
    return freq

def oneside_normalise(freq, X):
    N = len(X) // 2

    half_freqs = freq[:N]
    half_X = X[:N] / N  # take correct X and normalise

    return half_freqs, half_X

@ab_test(np.fft.fftn)
def fft_nd(x: np.ndarray):
    if not all(is_power_of_2(dim) for dim in x.shape):
        Exception("Dimesions of input x must be a power of 2")
    else:
        return _fft_nd(x)


def _fft_nd(x: np.ndarray):
    x = np.apply_along_axis(_fft_1d, 0, x)
    if x.ndim == 2:
        # if the number of dimensions is 2 then we can apply the fft along the columns
        x = np.apply_along_axis(_fft_1d, 1, x)

    return x

@ab_test(np.fft.fftshift)
def fft_shift(x: np.ndarray):
    N_x = x.shape[0]
    # swap first and second halfs of the array
    x = np.concatenate((x[N_x // 2 :], x[: N_x // 2]), axis=0)
    if x.ndim == 2:
        N_y = x.shape[1]
        # swap first and second halfs of the columns of the array
        x = np.concatenate((x[:, N_y // 2 :], x[:, : N_y // 2]), axis=1)

    return x

if __name__ == "__main__":
    N = 2**10
    M = 2**10

    x_range = (-10, 10)
    y_range = (-10, 10)

    fig, (ax1, ax2, ax3) = plt.subplots(3)

    x, y = np.meshgrid(np.linspace(*x_range, N), np.linspace(*y_range, M))

    gaussian = scipy.stats.multivariate_normal(
        mean=(0, 0), cov=[[1, 0], [0, 1]], allow_singular=False
    )
    pos = np.dstack((x, y))
    z = gaussian.pdf(pos)

    space_extent = np.concatenate((x_range, y_range))

    ax1.imshow(z, extent=space_extent)

    z = z.reshape(N, M)

    z_shifted = fft_shift(z)
    X = fft_nd(z_shifted)

    X_shift = np.fft.fftshift(X)
    
    sample_spacing_x = (x_range[1] - x_range[0])/N
    sample_spacing_y = (y_range[1] - y_range[0])/M

    magnitude_spectrum = np.abs(np.fft.fftshift(X))

    # freq_x_range = ()

    # freq_extent = np.concatenate((freq_x, freq_y))
    # ax2.imshow(magnitude_spectrum, extent=freq_extent)

    ax1.set_aspect('equal')
    ax2.set_aspect('equal')


    plt.show()

    X = fft_nd(x)
