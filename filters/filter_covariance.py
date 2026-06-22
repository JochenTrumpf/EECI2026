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
