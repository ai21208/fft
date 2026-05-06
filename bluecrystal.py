import os
os.environ [ "MKL_NUM_THREADS" ] = "1"
os.environ [ "NUMEXPR_NUM_THREADS" ] = "1"
os.environ [ "OMP_NUM_THREADS" ] = "1"

from fft_main import *
from benchmarking import *
import sys

if __name__ == "__main__":
    _, exponent, runs, cores = sys.argv
    bench_2d(fft_1d_iterative, 2**exponent, runs, [cores])