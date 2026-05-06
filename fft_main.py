import os
os.environ [ "MKL_NUM_THREADS" ] = "1"
os.environ [ "NUMEXPR_NUM_THREADS" ] = "1"
os.environ [ "OMP_NUM_THREADS" ] = "1"

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import numba as nb
import pandas as pd

from misc import *
import scipy

# from matplotlib import rc
# rc('font',**{'family':'serif','serif':['New Computer Modern']})
# rc('text', usetex=True)


def fft_1d_recursive(x):
    """
    Wrapper around _fft_1d_recursive function to check the input is valid when first called.
    The input size doesn't need to be checked for every recurive function call.
    """
    if not is_power_of_2(len(x)):
        raise ValueError("Input size must be a power of 2.")
    else:
        return _fft_1d_recursive(x)


# @ab_test(np.fft.fft)
@nb.jit(nopython=True, fastmath=True)
def _fft_1d_recursive(x: np.ndarray) -> np.ndarray:
    N: int = x.size

    if N == 1:
        return x.astype(np.complex64)
    # elif N <= 2**18:
    #     return _fft_1d_iterative(x)
    else:
        X_even = _fft_1d_recursive(x[::2]).astype(
            np.complex64
        )  # even elements recursively passed to fft
        X_odd = _fft_1d_recursive(x[1::2]).astype(
            np.complex64
        )  # odd elements recursively passed to fft

        n = np.arange(N, dtype=np.int64)
        exponentials = np.exp(-2j * np.pi * n / N).astype(np.complex64)

        # numba doesn't support np.concatenate
        X = X_even + exponentials[: int(N / 2)] * X_odd
        X = np.append(X, X_even + exponentials[int(N / 2) :] * X_odd)

        return X


@nb.jit(nopython=True)
def _bit_reverse(index: int, log_N: int) -> int:
    result = 0
    for _ in range(log_N):
        result = (result << 1) | (
            index & 1
        )  # Shift left and append the least significant bit
        index >>= 1  # Shift right to process the next bit
    return result


# @ab_test(np.fft.fft)
@nb.jit(nopython=True, fastmath=True)
def fft_1d_iterative(x: np.ndarray) -> np.ndarray:
    N = x.size
    if not is_power_of_2(N):
        raise ValueError("Input size must be a power of 2.")

    # Bit-reversal permutation
    X = x.astype(np.complex64)
    log_N = int(np.log2(N))
    for i in nb.prange(N):
        rev = _bit_reverse(i, log_N)
        if i < rev:  # Avoid double swaps
            X[i], X[rev] = X[rev], X[i]

    # Iterative FFT computation
    step = 2
    while step <= N:
        half_step = step // 2
        twiddle = np.exp(-2j * np.pi * np.arange(half_step) / step)
        for i in range(0, N, step):
            for j in range(half_step):
                # Butterfly operation
                a = X[i + j]
                b = twiddle[j] * X[i + j + half_step]
                X[i + j] = a + b
                X[i + j + half_step] = a - b
        step *= 2

    return X


@nb.njit()
def is_power_of_2(n):
    return (n & (n - 1) == 0) and n != 0


def fft_freq(sampling_rate: int, N: int) -> np.ndarray:
    freq = np.arange(N) * sampling_rate / N
    return freq


def oneside_normalise(freq, X):
    N = len(X) // 2

    half_freqs = freq[:N]
    half_X = X[:N] / N  # take correct X and normalise

    return half_freqs, half_X


# @ab_test(np.fft.fftn)
def fft_nd(x: np.ndarray):
    if not all(is_power_of_2(dim) for dim in x.shape):
        raise ValueError("Dimesions of input x must be a power of 2")
    else:
        x = x.astype(np.complex64)
        return _fft_nd_par(x)


def _fft_nd(x: np.ndarray):
    x = np.apply_along_axis(fft_1d_iterative, 0, x)
    if x.ndim == 2:
        # if the number of dimensions is 2 then we can apply the fft along the rows
        x = np.apply_along_axis(fft_1d_iterative, 1, x)

    return x


@nb.jit(nopython=True, parallel=True)
def _fft_nd_par(x: np.ndarray):
    # Apply the one dimensional fft along axis 0
    for i in nb.prange(x.shape[0]):
        x[i, :] = fft_1d_iterative(x[i, :])

    # if the input is 2-D then we apply the 1-D fft in the second axis too
    if x.ndim == 2:
        for j in nb.prange(x.shape[1]):  # Loop through columns
            x[:, j] = fft_1d_iterative(x[:, j])

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
