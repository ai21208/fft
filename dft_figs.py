from fft_main import *
import scipy


def simple_dft():
    duration = 2**2
    sr = 2**15  # sampling rate
    N = sr * duration

    t = np.linspace(0, duration, N)

    fig, (ax1, ax2) = plt.subplots(2)
    sig = gen_signal(t, [1, 7, 15], [3, 5, 0.5], 1)

    ax1.plot(t, sig, linewidth=0.5)
    ax1.set(xlabel="Time (s)", ylabel="Amplitude")

    # take only positive frequencies and normalize.
    X_oneside = fft_1d_iterative(sig)[: N // 2] / (N // 2)
    freq_oneside = fft_freq(sr, N)[: N // 2]

    ax2.plot(freq_oneside, abs(X_oneside), linewidth=1)
    ax2.set(
        xlim=[-1, 25],
        xlabel="Frequency (Hz)",
        ylabel="Amplitude",
        yticks=list(range(0, 6)),
    )

    plt.show()
    # fig.set_size_inches(5, 4)
    fig.tight_layout()

    fig.savefig("simple_dft.svg")


def simple_2d_dft():
    N = 2**12
    M = 2**12

    fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2)

    # Plot first result

    x, y = np.meshgrid(np.linspace(-N // 2, N // 2, N), np.linspace(-M // 2, M // 2, M))

    gaussian = scipy.stats.multivariate_normal(
        mean=(0, 0), cov=[[2**5, 0], [0, 2**5]], allow_singular=False
    )
    pos = np.dstack((x, y))
    z = gaussian.pdf(pos)

    im1 = ax1.imshow(z)
    plt.colorbar(im1, ax=ax1, label="Probability Density")
    ax1.set(
        title=r"Gaussian $\sigma = 2^5$",
        xlabel="x (m)",
        ylabel="y (m)",
    )
    ax1.set_xlim(N//2 - 2**8, N//2 + 2**8)
    ax1.set_ylim(M//2 - 2**8, M//2 + 2**8)

    z = z.reshape(N, M)

    X = fft_nd(z)
    X_shifted = fft_shift(X)

    mag_spectrum = abs(X_shifted)  # magnitude spectrum
    # log just to show the magnitude spectrum more clearly
    # log_spectrum = np.log1p(mag_spectrum) # add one to avoid log(0)

    im2 = ax2.imshow(mag_spectrum)
    plt.colorbar(im2, ax=ax2, label=r"Magnitude Spectrum $|X|$")
    ax2.set(
        title="Magnitude Spectrum",
        xlabel="Spatial Frequency (cycles/m)",
        ylabel="Spatial Frequency (cycles/m)",
    )
    ax2.set_xlim(N//2 - 2**8, N//2 + 2**8)
    ax2.set_ylim(M//2 - 2**8, M//2 + 2**8)

    # plot second result

    x, y = np.meshgrid(np.linspace(-N // 2, N // 2, N), np.linspace(-M // 2, M // 2, M))

    gaussian = scipy.stats.multivariate_normal(
        mean=(0, 0), cov=[[2**13, 0], [0, 2**13]], allow_singular=False
    )
    pos = np.dstack((x, y))
    z = gaussian.pdf(pos)

    im3 = ax3.imshow(z)
    plt.colorbar(im3, ax=ax3, label="Probability Density")
    ax3.set(
        title=r"Gaussian $\sigma = 2^{13}$",
        xlabel="x (m)",
        ylabel="y (m)",
    )
    ax3.set_xlim(N//2 - 2**8, N//2 + 2**8)
    ax3.set_ylim(M//2 - 2**8, M//2 + 2**8)
    
    z = z.reshape(N, M)

    X = fft_nd(z)
    X_shifted = fft_shift(X)

    mag_spectrum = abs(X_shifted)  # magnitude spectrum
    # log just to show the magnitude spectrum more clearly
    # log_spectrum = np.log1p(mag_spectrum) # add one to avoid log(0)

    im4 = ax4.imshow(mag_spectrum)
    plt.colorbar(im4, ax=ax4, label=r"Magnitude Spectrum $|X|$")
    ax4.set(
        title="Magnitude Spectrum",
        xlabel="Spatial Frequency (cycles/m)",
        ylabel="Spatial Frequency (cycles/m)",
    )
    ax4.set_xlim(N//2 - 2**8, N//2 + 2**8)
    ax4.set_ylim(M//2 - 2**8, M//2 + 2**8)

    ax1.set_aspect("equal")
    ax2.set_aspect("equal")
    ax3.set_aspect("equal")
    ax4.set_aspect("equal")

    plt.tight_layout()
    plt.show()

    fig.savefig("fig.svg")

simple_2d_dft()
