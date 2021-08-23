# -*- coding: utf-8 -*-
"""
Created on Sat June  17 15:17:59 2021
@author: allisonoge
"""
import sys

import numpy as np
from scipy import fftpack, signal, stats

# from current directory
from estnoise_ms import *
from M_D import M

_D = 96
_U = 8
_V = _D//_U
_Md = M(_D)
_Mv = M(_V)


class stft:
    def __init__(self) -> None:
        self.shift = None

    def compute(self, sig, samp_rate, nfft, window_time=0.03, overlap=0.5):
        """A short time fourier transform of a time signal
        @signal -- time signal
        @samp_rate -- sample rate of the signal
        @nff -- FFT length
        @window_time -- window time
        @overlap -- overlap"""

        if window_time > len(sig)/samp_rate:
            sys.stderr.write(
                "Window time should be less than the time length of signal")
            sys.exit(1)
        window_length = int(samp_rate*window_time)
        self.shift = int((1-overlap)*window_length)
        window = signal.blackman(window_length)
        return np.array([np.abs(fftpack.fft(window*sig[i:i+window_length], nfft)[:nfft//2]) for i in range(0, len(sig)-window_length, self.shift)])

    def get_shift(self):
        return self.shift


def gen_wgnoise(sig):
    # snr of 40 dB
    snr_lin = 10**(40/10)
    sig_pow = sum([s**2 for s in sig])/len(sig)
    amp = np.sqrt(sig_pow/snr_lin)
    return stats.norm(0, amp)


class estnoisems:
    def __init__(self, nfft, niteration) -> None:
        self.noise_est = np.ones(nfft)*np.inf
        self.alpha_vec = np.ones(nfft)*0.96
        self.alpbuff = np.ones((niteration, nfft))*0.96
        self.psd_vec = np.ones(nfft)
        self.psdbuff = np.ones((niteration, nfft))
        self.actmin_vec = np.ones(nfft)*np.inf
        self.actmin_sub_vec = np.ones(nfft)*np.inf
        self.actbuff = np.ones((_U, nfft))*np.inf
        self.subwc = _V

    def compute(self, Y_vec, niteration):
        """Estimation of the noise based on minimum statistics"""
        self.psd_vec = Y_vec[0, :]**2
        self.noise_est = self.psd_vec
        alpha_corr = 1
        fmoment = self.psd_vec
        smoment = self.psd_vec**2
        lmin_flag = np.zeros(len(self.psd_vec))
        ibuf = 0
        x = np.zeros((niteration, len(self.psd_vec)))
        for n in range(niteration):
            Y_vec_n = Y_vec[n, :]  # consider only a frame
            # compute tilda  correction factor
            talpha_corr = 1/(1 + (sum(self.psd_vec)/sum(Y_vec_n**2) - 1)**2)
            tmp = np.array([talpha_corr])
            tmp[tmp < 0.7] = 0.7
            alpha_corr = 0.7 * alpha_corr + 0.3 * tmp
            self.alpha_vec = 0.96 * alpha_corr / \
                (1 + (self.psd_vec/self.noise_est - 1)
                 ** 2)  # compute the smoothing factor
            self.alpbuff[n, :] = self.alpha_vec     # save all alpha values
            self.psd_vec = self.alpha_vec * self.psd_vec + \
                (1-self.alpha_vec) * Y_vec_n**2     # compute the smoothed psd
            self.psdbuff[n, :] = self.psd_vec
            # compute beta and P_var_vec
            beta_vec = min_complex(self.alpha_vec ** 2, np.array([0.8]))
            fmoment = beta_vec * fmoment + \
                (1 - beta_vec) * self.psd_vec
            smoment = beta_vec * smoment + \
                (1 - beta_vec) * self.psd_vec ** 2
            var = smoment - fmoment ** 2
            # compute the DOF
            DOF_vec = max_complex(
                (2 * self.noise_est ** 2) / var, np.array([2.0]))
            # compute the tDOF for windows
            tDOF_vec = (DOF_vec - 2 * _Md.get_m) / (1 - _Md.get_m)
            tDOF_sub_vec = (DOF_vec - 2 * _Mv.get_m) / (1 - _Mv.get_m)
            # compute the bias
            bias_vec = 1 + ((_D - 1) * 2 / tDOF_vec)
            bias_sub_vec = 1 + ((_V - 1) * 2 / tDOF_sub_vec)
            # compute q_inv_mean
            Q_inv_mean = sum(1/DOF_vec)/len(Y_vec)
            # compute b_corr
            B_corr = 1 + (2.12*np.sqrt(Q_inv_mean))
            k_mod = self.psd_vec*bias_vec*B_corr < self.actmin_vec
            if any(k_mod):
                self.actmin_vec[k_mod] = self.psd_vec[k_mod] * \
                    bias_vec[k_mod]*B_corr
                self.actmin_sub_vec[k_mod] = self.psd_vec[k_mod] * \
                    bias_sub_vec[k_mod]*B_corr
            if self.subwc > 0 and self.subwc < _V:
                lmin_flag = np.logical_or(k_mod, lmin_flag)
                P_min_u = min_complex(self.actmin_vec, P_min_u)
                self.noise_est = P_min_u.copy()
                self.subwc += 1
            else:
                if self.subwc >= _V:
                    lmin_flag = np.logical_and(k_mod, lmin_flag)
                    # uses buffer for storage of the past u frames
                    ibuf = 1+(ibuf % _U)    # increment pointer
                    self.actbuff[ibuf-1, :] = self.actmin_vec.copy()
                    P_min_u = min_complex_mat(self.actbuff)
                    if Q_inv_mean < 0.03:
                        noise_slope_max = 10**(8/10)
                    elif Q_inv_mean < 0.05:
                        noise_slope_max = 10**(4/10)
                    elif Q_inv_mean < 0.06:
                        noise_slope_max = 10**(2/10)
                    else:
                        noise_slope_max = 10**(1.2/10)
                    lmin = np.logical_and(np.logical_and(np.logical_and(lmin_flag, np.logical_not(
                        k_mod)), self.actmin_sub_vec < (noise_slope_max*P_min_u)), self.actmin_sub_vec < P_min_u)
                    if any(lmin):
                        P_min_u[lmin] = self.actmin_sub_vec[lmin]
                        # replace all previously stored actmin with actmin_sub
                        self.actbuff[:, lmin] = np.ones((_U, 1))*P_min_u[lmin]
                    lmin_flag[:] = 0
                    self.actmin_vec[:] = np.Inf
                    self.actmin_sub_vec[:] = np.Inf
                    self.subwc = 1
            x[n, :] = self.noise_est
        return x

    def get_alpha(self):
        return self.alpbuff

    def get_smoothed(self):
        return self.psdbuff
