import numpy as np

class filter_covariance():
    """
    This is a container class for the covariance variables of a filter

    Callable interface:
     - copy()
        + return a deep copy of self   
    """

    def __init__(self):
        pass

    def copy(self):
        return self.__class__()

class INS_filter_covariance(filter_covariance):

    def __init__(self, Pr: np.typing.NDArray):
        self.Pr = Pr

    def copy(self):
        return self.__class__(self.Pr)

class INS_EqF_covariance(filter_covariance):

    def __init__(self, P: np.typing.NDArray):
        self._P = P                     # 15x15 covariance for (p, v, q, ab, wb)                     
        self.Pr = self._P[0:9, 0:9]     # (marginalized) covariance for (p, v, q)

    @property
    def P(self):
        return self._P

    @P.setter
    def P(self, value: np.typing.NDArray):
        self._P = value
        self.Pr = self._P[0:9, 0:9]     # (marginalized) covariance

    def copy(self):
        return self.__class__(self._P)
