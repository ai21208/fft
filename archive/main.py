import numpy as np
import matplotlib.pyplot as plt


def cas(x: np.ndarray) -> np.ndarray:
    """
    cas(x) = cos(x) + sin(x) = sqrt(2)cos(x-pi/4)
    """
    return np.sqrt(2) * np.cos(x - np.pi / 4)


# Real to Real
def dht(x):
    N = len(x)
    H = np.zeros(N)
    
    for k in range(N):
        angle = 2 * np.pi * np.arange(N) * k / N
        H[k] = np.sum(x * cas(angle))
    
    return H

def freq(n, Fs):
    return np.linspace(0, Fs, n)

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

X = dht(noisy_signal) # np.fft.fft(noisy_signal, N)[:N//2]
# X = np.abs(X)
# X /= X.max()
f = freq(len(X), sample_rate)

fig, ax = plt.subplots()
ax.plot(t, dht(X) - noisy_signal)
# ax.set_xlim(-20)

plt.show()
