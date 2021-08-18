import sys

import numpy as np

_D = np.array([1, 2, 5, 8, 10, 15, 20, 30, 40, 60, 80, 120, 140, 160])
_M = np.array([0, 0.26, 0.48, 0.58, 0.61, 0.668, 0.705,
               0.762, 0.8, 0.841, 0.865, 0.89, 0.9, 0.91])


class M:
    def __init__(self, d):
        self.d = _D
        self.m = _M
        if np.where(self.d == d)[0]:
            self.get_m = float(self.m[np.where(self.d == d)[0]])
        else:
            # intrapolation
            intra_index_1 = intra_index_2 = None
            for i in range(len(self.d)):
                if d < self.d[i]:
                    intra_index_1 = i-1
                    intra_index_2 = i
                    break
            if not (intra_index_1 and intra_index_2):
                sys.stderr.write("WARNING: parameter D is out of range!")
                sys.exit(1)
            self._intrapolate(d, intra_index_1, intra_index_2)

    def _intrapolate(self, d, ind_1, ind_2):
        self.get_m = ((self.m[ind_2] - self.m[ind_1])*(d - self.d[ind_1])
                      )/(self.d[ind_2] - self.d[ind_1]) + self.m[ind_1]

    def get_m(self):
        return self.get_m
