import numpy as np
import matplotlib.pyplot as plt
# from numba import jit
import time
import numpy.typing as npt


# @jit(nopython=True)
def dft(x):
    N = len(x)
    X = np.zeros(N, dtype=np.complex128)  # Output array for the DFT results
    for k in range(N):  # For each output element
        sum_val = 0.0j  # Start with a complex zero
        for n in range(N):  # Sum over the input sequence
            angle = -2j * np.pi * k * n / N
            sum_val += x[n] * np.exp(angle)
        X[k] = sum_val
    return X


# @jit(nopython=True)
def dft2(x):
    N = len(x)
    X = np.zeros(N, dtype=np.complex128)  # Output array for the DFT results
    for k in range(N):  # For each output element
        n = np.arange(N)
        angle = -2j * np.pi * k * n / N
        X[k] = (x * np.exp(angle)).sum()
    return X

def fft(x):
    N = len(x)
    X = np.zeros(N, dtype=np.complex128)  # Output array for the DFT results
    for k in range(N):  # For each output element
        out = F_k(x, k, 0)
        print(out)
        X[k] = out

    print(F_k(x, 100, 0))
    return X


def F_k(x, k, total):
    """
    x: input function points
    k
    """
    length = len(x)
    print(k)
    if length % 2 == 0:
        return F_k(x[::2], k, total) + F_k(x[1::2], k, total) * np.exp(2 * np.pi * 1j / N)
    else:
        j = np.arange(length)
        exponent = -2j * np.pi * k * j / length
        return (x * np.exp(exponent)).sum()


def timer(func):
    start = time.time()
    return_val = func()
    print(f"Function took {time.time() - start} seconds")
    return return_val


duration = 10
sample_rate = 1024
num_samples = sample_rate * duration

t = np.linspace(0, 10, num_samples, False)
N = num_samples

frequencies = np.array([5, 20, 50])
amplitudes = np.array([10, 5, 3])

clean_signal = sum(
    amplitude * np.sin(2 * np.pi * frequency * t)
    for amplitude, frequency in zip(amplitudes, frequencies)
)
noisy_signal = clean_signal + np.random.normal(0, 5, num_samples)

fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(t, noisy_signal)

noisy_freq = timer(lambda: dft2(noisy_signal[1:7]))
print(noisy_freq)

noisy_freq2 = timer(lambda: fft(noisy_signal[1:7]))
print(noisy_freq2)


ax2.plot(np.linspace(0, sample_rate // 2, N // 2), abs(noisy_freq)[: N // 2])
ax3.plot(np.linspace(0, sample_rate // 2, N // 2), abs(noisy_freq2)[: N // 2])
plt.show()
