import numpy as np


class KF:
    """
    Instantiates with the following properties:

    initial_x: initial location
    initial_v: initial velocity
    """

    def __init__(self, initial_x, initial_v) -> None:
        # mean of state gaussian random variable
        self.x = np.array([initial_x, initial_v])

        # covariance of state gaussian random variable
        self.p = np.eye(2)


kf = KF(initial_x=0.2, initial_v=0.5)
