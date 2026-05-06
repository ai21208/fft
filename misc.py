import time
import numpy as np
import timeit
import functools

"""
Times the execution time of a function
"""
def timer(func):
    start = time.time()
    return_val = func()
    print(f"Function {func.__name__} took {time.time() - start} seconds")
    return return_val

"""
Decorator to check the output against library functions that are known to be correct.
Should not be used when benchmarking.

Example usage:

@ab_test(np.fft.fft)
def my_dft(x):
    ...
"""
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
def benchmark(code: str, globals, number: int) -> np.ndarray:
    # setup: allow function to complile first time
    timeit.timeit(code, globals=globals, number=1)

    times = np.array(
        [timeit.timeit(code, globals=globals, number=1) for _ in range(number)]
    )
    return times

"""
Generates a signal that is a superposition of sin waves and a gaussian noise functions.
"""
def gen_signal(
    t: np.ndarray, frequencies: np.ndarray, amplitudes: np.ndarray, noise: float
) -> np.ndarray:
    N = len(t)
    clean_signal = sum(
        amplitude * np.sin(2 * np.pi * frequency * t)
        for amplitude, frequency in zip(amplitudes, frequencies)
    )
    noisy_signal = clean_signal + np.random.normal(0, noise, N)

    return noisy_signal

"""
Create a dictionary of some basic statistics of a numpy array
"""
def describe(array: np.ndarray) -> dict:
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