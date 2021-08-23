import sys
from os.path import dirname
from os.path import join as pjoin

import matplotlib as mpl
import numpy as np
import scipy.io
from matplotlib import pyplot as plt
from scipy import fftpack, signal, stats
from scipy.io import wavfile
from scipy.signal.spectral import spectrogram, stft

import estnoise_ms
# from current directory
import ms_estnoise
from estnoise_ms import estnoisem
from min_stats_est import MS


def subsample_plot(sig, samp_rate, window_time=0.032, nfft=1024):
    my_stft = ms_estnoise.stft()
    return np.transpose(my_stft.compute(sig, samp_rate, nfft, window_time, 0.5)), my_stft.compute(sig, samp_rate, nfft, window_time, 0.5).shape[0]


def noise_estimation(spectrogram):
    """Noise Estimation of the power spectral density based on the minimum statistics"""
    (niteration, nfft) = spectrogram.shape
    estimator = ms_estnoise.estnoisems(nfft, niteration)
    noise_est = estimator.compute(spectrogram, niteration)
    smoothed = estimator.get_smoothed()
    alpha = estimator.get_alpha()
    return smoothed, noise_est, alpha


def label_ax(axes, xlabel, ylabel):
    """"an util function for labelling the axes of the figure"""
    fs = 16
    axes.set_xlabel(xlabel, fontsize=fs)
    axes.set_ylabel(ylabel, fontsize=fs)


def visualize(sig, spectrogram, time_sig, time_freq, freq, filename="visualize"):
    # create grid for data presentation
    fig = plt.figure(figsize=(16, 8))
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((4, 1), (2, 0))
    # ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)

    label_ax(ax1, 'time (s)', 'frequency (Hz)')
    label_ax(ax2, 'time (s)', 'Relative gain (dB)')

    t_mesh, f_mesh = np.meshgrid(time_freq, freq)
    spectrogram = 20*np.log(spectrogram/np.max(spectrogram))
    norm = mpl.colors.Normalize(np.min(spectrogram), np.max(spectrogram) + 3)

    plot = ax1.pcolor(t_mesh, f_mesh, spectrogram, norm=norm, cmap=mpl.cm.jet)

    ax1.axis('tight')

    ax2.plot(time_sig, sig)

    cb = fig.colorbar(plot, ax=ax1)
    cb.set_label('Relative Gain', fontsize=14)

    plt.tight_layout()
    print("Saving figure!")
    plt.savefig(filename + ".png")

    plt.show()


def smoothing_val(sig, samp_rate, spectrogram, tlength, time, domain=0, filename="smoothing validation"):
    """Validates the smoothing factor of the algorithm"""
    x_axis = ['time (secs)', 'frequency (Hz)']
    (niteration, nfft) = spectrogram.shape
    (smoothed, estimated_noise, alpha) = noise_estimation(spectrogram)
    fig = plt.figure(figsize=(16, 8))
    ax1 = plt.subplot2grid((3, 1), (0, 0))
    ax2 = plt.subplot2grid((3, 1), (1, 0))
    ax3 = plt.subplot2grid((3, 1), (2, 0))

    ftime = np.linspace(0, tlength, niteration)

    if not domain:
        label_ax(ax1, x_axis[domain], ' Norm magnitude')
        label_ax(ax2, x_axis[domain], 'speech signal')
        label_ax(ax3, x_axis[domain], 'smoothing fac')

        freq_ind = 2500*nfft*2//samp_rate
        fig.suptitle(
            "Noise estimation at 25ms window time using a 50% overlap")
        ax1.plot(
            ftime, 20*np.log(spectrogram[:, freq_ind]), label='periodogram')
        ax1.plot(
            ftime, 10*np.log(smoothed[:, freq_ind]), label='smoothed periodogram')
        #ax1.plot(ftime, estimated_noise[:, freq_ind], label='Estimated noise')

        ax2.plot(time, sig)

        ax3.plot(ftime, alpha[:, freq_ind])

        ax1.legend()
    else:
        label_ax(ax1, x_axis[domain], ' Relative Gain (dB)')
        label_ax(ax2, x_axis[domain], 'smoothing fac')

        time_ind = int(9.0 * niteration // time[-1])
        freq = np.linspace(0, samp_rate/2, nfft//2)

        ax1.plot(freq, 20*np.log(spectrogram[time_ind, :]))
        ax2.plot(freq, alpha[time_ind, :])

    plt.tight_layout()
    print("Saving figure!")
    plt.savefig(filename + ".png")
    plt.show()


def validation(spectrogram, true_noise, estimated_noise, stime, samp_rate, filename="est_validation"):
    """Validates the algorithm using the minimum squared error analysis to measure the accuracy of the estimation
    Produces two subplots of the noise estimate alongside the signal periodogram and the true noise versus the estimated noise"""
    # create grid for data presentation
    fig = plt.figure(figsize=(16, 8))
    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0))

    label_ax(ax1, 'time (seconds)', "Norm magnitude")
    label_ax(ax2, "time (seconds)", "Norm magnitude")

    (niteration, nfft) = spectrogram.shape
    time = np.linspace(0, stime, niteration)

    # mse = mean of the variance of the the true and estimated noise data set
    var = np.sum((estimated_noise - true_noise) ** 2,
                 axis=1) / np.sum(true_noise, axis=1)
    # normalize mse array
    ref = var.copy()
    var = (var - ref.min()) / (ref.max() - ref.min())
    mse = np.mean(var)

    estimated_noise = 10*np.log(estimated_noise)
    true_noise = 20*np.log(true_noise)
    spectrogram = 20*np.log(spectrogram)

    freq_ind = 2500 * nfft * 2 // samp_rate
    fig.suptitle(
        'Objective measures of the estimated noise using mean squared error (MSE). The MSE calculated is {}'.format(mse))
    ax1.plot(time, spectrogram[:, freq_ind], label='Periodogram')
    ax1.plot(time, estimated_noise[:, freq_ind], label='Estimated noise')

    ax2.plot(time, true_noise[:, freq_ind], label='True noise')
    ax2.plot(time, estimated_noise[:, freq_ind], label='Estimated noise')

    ax1.legend()
    ax2.legend()

    plt.tight_layout()
    print("Saving figure!")
    plt.savefig(filename + ".png")

    plt.show()


if __name__ == "__main__":
    data_dir = pjoin('sounds')
    wav_fname = pjoin(data_dir, 'test_sound.wav')
    nfft = 512
    samplerate, data = wavfile.read(wav_fname)
    length = data.shape[0] / samplerate
    print(f"seconds of signal = {length}s")
    a = data.T[0]
    # this is 8-bit track, b is now normalized on [-1,1)
    b = [(ele/2**8.)*2-1 for ele in a]

    # add noise
    wgnoise = ms_estnoise.gen_wgnoise(b)
    noise_sig = wgnoise.rvs(data.shape[0])
    noisy_sig = b + noise_sig

    # short-term fourier transform
    my_stft = ms_estnoise.stft()
    spectrogram = my_stft.compute(noisy_sig, samplerate, nfft, nfft/samplerate)
    true_noise = my_stft.compute(noise_sig, samplerate, nfft, nfft/samplerate)

    # noise estimation
    result = noise_estimation(spectrogram)

    # visualize spectrogram
    time_sig = np.linspace(0., length, data.shape[0])
    # visualize(b, target_data, time_sig, time_freq, freq, 'periodogram')

    # Validate algorithm
    # smoothing_val(noisy_sig, samplerate, spectrogram, length,
    #               time_sig)
    validation(spectrogram, true_noise, result[1], length, samplerate)
