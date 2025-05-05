import numpy as np


class EEP:
    """
    An elliptical error probabl distribution
    """
    def __init__(self, cov: np.array, cx: float = 0.0, cy: float = 0.0):
        """
        Args:
            cov: the covariance matrix defining the distribution
            cx: the x center of the distribution
            cy: the y center of the distribution
        """
        self._cov_mat = cov
        self._mean_eep_x = cx
        self._mean_eep_y = cy

    def sample(self, scale: float = 1.0) -> tuple[float, float]:
        """
        Draw a sample from the EEP distribution
        Args:
            scale: the scale to sample at. Allows conversion from cm to m, for example

        Returns: sample x, sample y
        """
        mx = self._mean_eep_x * scale
        my = self._mean_eep_y * scale
        cov = self._cov_mat * scale
        # this has been tested and shown to produce an eep distribution that matches the expected.
        sample = np.random.multivariate_normal([mx, my], cov).T
        return float(sample[0]), float(sample[1])
