# !usr/bin/python
"""
Created on Wed Aug 25 13:45 2021
@author: allisonOge
"""

import numpy as np


def estnoisefc(spectrogram, info=dict({"chan_bw": 500e3, "samp_rate": 6e6, "nchan": 6, "fchan": 4})):
    """estimated the noise of the measured band using the noise level of a free channel in the band"""
    (niteration, nfft) = spectrogram.shape
    chan_bw = nfft * info['chan_bw'] // info['samp_rate']

    x = np.ones((niteration, nfft))
    for n in range(niteration):
        Y_vec = spectrogram[n, :]
        # match dimensions for odd division of spectrum band
        Y_vec = Y_vec[:info['nchan']*int(chan_bw)]

        Y_vec = Y_vec.reshape((info['nchan'], int(chan_bw)))
        noise_est = np.ones(
            nfft) * sum([y**2 for y in Y_vec[info['fchan'], :]])/nfft
        x[n, :] = noise_est
    return x


def estnoise80(spectrogram):
    "estimate the noise of the measured band using only the 20th percentile of the given samples in the band"
    (niteration, nfft) = spectrogram.shape

    x = np.ones((niteration, nfft))
    for n in range(niteration):
        sorted_spectrogram = np.sort(spectrogram)
        # look up the 20th percentile
        ind_20 = int(0.2 * nfft)
        noise_est = sum([y**2 for y in sorted_spectrogram[n, :ind_20]])/ind_20
        x[n, :] = noise_est
    return x
