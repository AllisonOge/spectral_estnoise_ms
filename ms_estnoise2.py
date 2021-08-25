# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 17:09 2021
@author: allisonOge
"""

import math

import numpy as np

# from current directory
from estnoise_ms import *
from M_D import M

_D = 64  # 1 of 8 of the nfft (512)
_Md = M(_D)


class estnoisems:
    """A noise estimation using minimum statistics with concerntration towards the frequency domain. A nuance implementation of the existing algorithm proposed by Martin"""

    def __init__(self, nfft, niteration):
        """initialize constructor"""
        self.__noise_est = np.ones(nfft) * np.Inf
        self.__last_psd = np.Inf
        self.__psd_vec = np.ones(nfft) * np.Inf
        self.__alpha_c = 0.7
        self.__alpha = np.ones(nfft) * 0.96
        self.__alpha_buff = np.ones((niteration, nfft)) * np.Inf
        self.__psd_buff = np.ones((niteration, nfft)) * np.Inf

    def compute(self, Y_mat):
        """compute the noise estimate"""
        # Y_mat is a matrix so we consider the first frame
        Y_vec = Y_mat[0, :]
        (niteration, nfft) = Y_mat.shape
        self.__psd_vec = Y_vec**2
        self.__last_psd = self.__psd_vec[0]
        self.__noise_est = self.__psd_vec
        fmoment = self.__psd_vec
        smoment = self.__psd_vec**2
        x = np.ones((niteration, nfft))
        alpha_max = 0.96  # upper limit on alpha_var

        for n in range(niteration):
            # compute the correction factor
            tmp = sum(self.__psd_vec) / sum(Y_vec ** 2) - 1
            corfac_term = 0.3 * max(1 / (1 + tmp ** 2), 0.7)
            self.__alpha_c = 0.7 * self.__alpha_c + corfac_term
            self.__alpha = alpha_max * self.__alpha_c / \
                (1 + (self.__psd_vec / self.__noise_est - 1) ** 2)
            self.__alpha_buff[n, :] = self.__alpha

            # compute beta for estimating psd variance
            beta_vec = min_complex(self.__alpha ** 2, np.array([0.8]))
            fmoment = beta_vec * fmoment + \
                (1 - beta_vec) * self.__psd_vec
            smoment = beta_vec * smoment + \
                (1 - beta_vec) * self.__psd_vec ** 2
            var = smoment - fmoment ** 2

            # self.__psd_vec = self.__alpha * \
            #     self.__psd_vec + (1 - self.__alpha) * Y_vec**2
            # loop through frame (frequency elements)
            for k in range(nfft):
                self.__last_psd = self.__alpha[k] * self.__last_psd + \
                    (1 - self.__alpha[k]) * Y_vec[k] ** 2
                self.__psd_vec[k] = self.__last_psd
            self.__psd_buff[n, :] = self.__psd_vec
            # compute the DOF
            DOF_vec = max_complex(
                (2 * self.__noise_est ** 2) / var, np.array([2.0]))
            # compute the tDOF for windows
            tDOF_vec = (DOF_vec - 2 * _Md.get_m) / (1 - _Md.get_m)
            # compute mean of the inverse of the DOF
            Q_inv_mean = sum(1/tDOF_vec) / nfft

            # compute the bias and correction factor
            bias = 1 + ((_D - 1) * 2 / tDOF_vec)
            bias_c = 1 + 2.12 * np.sqrt(Q_inv_mean)

            # db constraints between 1.2 - 8dB
            if Q_inv_mean < 0.03:
                noise_slope_max = 10**(8/10)
            elif Q_inv_mean < 0.05:
                noise_slope_max = 10**(4/10)
            elif Q_inv_mean < 0.06:
                noise_slope_max = 10**(2/10)
            else:
                noise_slope_max = 10**(1.2/10)

            # intialization of actmin
            self.__actmin = min(self.__psd_vec[:_D])
            for k in range(nfft-_D+1):
                # to find the minimum in the frequency domain, the min in a window of length D is compared with the minimum of the updated window before updating the noise estimate
                if self.__actmin * \
                    noise_slope_max > min(
                        self.__psd_vec[k:k+_D] * bias[k:k+_D] * bias_c):
                    self.__actmin = min(
                        self.__psd_vec[k:k+_D] * bias[k:k+_D] * bias_c)
                self.__noise_est[k] = self.__actmin

            x[n, :] = self.__noise_est
        return x

    def get_alpha(self):
        return self.__alpha_buff

    def get_smoothed(self):
        return self.__psd_buff
