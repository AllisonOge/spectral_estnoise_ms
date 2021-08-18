# spectral noise estimation in python 3
#!\usr\bin3

import numpy as np

# from current directory
from M_D import M

_D = 96
_U = 8
_V = _D//_U
_M = M(_D)
_M_sub = M(_V)
# print(_M.get_m)


class MS_2:
    def __init__(self, Y_vec, preset):
        L = len(Y_vec)
        self.subwc = _V
        self.actmin_vec = np.ones(L) * preset
        self.actmin_sub_vec = np.ones(_V) * preset
        self.psd_vec = abs(Y_vec)**2
        self.alpha_corr = 1/(1 + (sum(self.psd_vec)/sum(Y_vec))**2)
        self.alpha_vec = np.zeros(L)
        self.beta_vec = np.zeros(L)
        self.fmoment = self.psd_vec
        self.smoment = self.psd_vec**2
        self.var = np.ones(L)
        self.sigma_noise_vec = np.ones(L)

    def get_noise_est(self, Y_vec):
        # compute the tilda correction factor
        talpha_corr = 1/(1 + (sum(self.psd_vec)/sum(Y_vec))**2)
        lmin_flag = bias_vec = bias_sub_vec = DOF_vec = tDOF_sub_vec = tDOF_vec = np.ones(
            len(Y_vec))
        for k in range(len(Y_vec)):
            # compute the correction factor
            self.alpha_corr = 0.7 * self.alpha_corr + \
                0.3 * np.max(0.7, talpha_corr)
            # compute the smoothing factor
            self.alpha_vec[k] = 0.98 * self.alpha_corr / \
                (1 + (self.psd_vec[k]/(self.sigma_noise_vec[k]))**2)
            # compute the periodogram
            self.psd_vec[k] = self.alpha_vec[k] * self.psd_vec[k] + \
                (1-self.alpha_vec[k]) * abs(Y_vec[k])**2
            # compute beta and P_var_vec
            self.beta_vec[k] = min(self.alpha_vec[k]**2, 0.8)
            self.fmoment = self.beta_vec[k]*self.fmoment + \
                (1-self.beta_vec[k])*self.psd_vec[k]
            self.smoment = self.beta_vec[k]*self.smoment[k] + \
                (1-self.beta_vec[k])*self.psd_vec[k]**2
            self.var[k] = self.smoment[k] - self.fmoment[k]**2
            # compute the DOF
            DOF_vec[k] = (2 * self.noise_est[k]**2) / self.var[k]
            # compute the tDOF
            tDOF_vec[k] = (DOF_vec[k] - 2 * _M.get_m)/(1 - _M.get_m)
            tDOF_sub_vec[k] = (DOF_vec[k] - 2 * _M_sub.get_m) / \
                (1 - _M_sub.get_m)
            # compute the bias
            bias_vec[k] = 1 + ((_D - 1)*2/tDOF_vec[k])
            bias_sub_vec[k] = 1 + ((_V - 1)*2/tDOF_sub_vec[k])
            # compute q_inv_mean
            Q_inv_mean = sum(1/tDOF_vec)/len(Y_vec)
            # compute b_corr
            B_corr = 1 + (2.12*np.sqrt(Q_inv_mean))
            k_mod = np.ones(len(Y_vec))
            if (self.psd_vec[k]*bias_vec[k]*B_corr < self.actmin_vec[k]):
                self.actmin_vec[k] = self.psd_vec[k]*bias_vec[k]*B_corr
                self.actmin_sub_vec[k] = self.psd_vec[k]*bias_sub_vec[k]*B_corr
                k_mod[k] = 1

            if self.subwc == _V:
                if k_mod[k] == 1:
                    lmin_flag[k] = 0
                # store actmin
                storage = self.actmin_vec
                P_min_u = min(storage)
                if Q_inv_mean < 0.03:
                    noise_slope_max = 8
                elif Q_inv_mean < 0.05:
                    noise_slope_max = 4
                elif Q_inv_mean < 0.06:
                    noise_slope_max = 2
                else:
                    noise_slope_max = 1.2
                if lmin_flag[k] and self.actmin_vec[k] < noise_slope_max*P_min_u and actmin_sub_vec[k] > P_min_u:
                    self.actmin_vec = actmin_sub_vec
                lmin_flag[k] = 0
                self.subwc = 1
                self.actmin_vec = max(self.actmin_vec)*np.ones(len(Y_vec))
                actmin_sub_vec = max(actmin_sub_vec)*np.ones(len(Y_vec))
            else:
                if self.subwc > 1:
                    if k_mod[k] == 1:
                        lmin_flag[k] = 1
                    self.sigma_noise_vec[k] = min(actmin_sub_vec[k], P_min_u)
                    P_min_u = self.sigma_noise_vec[k]
                self.subwc += 1
