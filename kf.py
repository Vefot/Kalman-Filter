import numpy as np


class KF:
    """
    Instantiates with the following properties:

    initial_x: initial location
    initial_v: initial velocity
    """

    def __init__(self, initial_x: float, initial_v: float) -> None:
        # mean of state gaussian random variable
        self.x = np.array([initial_x, initial_v])

        # covariance of state gaussian random variable
        self.p = np.eye(2)

    @property
    def pos(self) -> float:
        return self.x[0]

    @property
    def vel(self) -> float:
        return self.x[1]


kf = KF(initial_x=0.2, initial_v=0.5)
