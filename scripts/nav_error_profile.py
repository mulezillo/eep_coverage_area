import numpy as np


class NavErrorProfile:
    """
    A nav profile defined by an elliptical error distribution
    """
    def __init__(self, cov: np.array, cx: float = 0.0, cy: float = 0.0):
        self._cov_mat = cov
        self._mean_eep_x = cx
        self._mean_eep_y = cy

    def sample(self, scale: float = 1.0):
        mx = self._mean_eep_x * scale
        my = self._mean_eep_y * scale
        cov = self._cov_mat * scale
        # this has been tested and shown to produce an eep distribution that matches the expected.
        sample = np.random.multivariate_normal([mx, my], cov).T
        return float(sample[0]), float(sample[1])
