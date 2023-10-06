import numpy as np


class KF:
    """
    Instantiates with the following properties:

    initial_x: initial location
    initial_v: initial velocity
    """

    def __init__(
        self, initial_x: float, initial_v: float, accel_variance: float
    ) -> None:
        # mean of state gaussian random variable
        self._x = np.array([initial_x, initial_v])
        self._accel_variance = accel_variance

        # covariance matrix - represents uncertainty in the state
        self._P = np.eye(2)  # eye returns a 2x2 identity matrix

    def predict(self, dt: float) -> None:
        """
        Predicts the state vector forward in time by dt (delta time) seconds
        """
        # x = F * x
        # p = F * p * F^T  * a

        # state transition matrix
        F = np.array(([1, dt], [0, 1]))

        # calculate the new state by multiplying state transition matrix with current x
        new_x = F.dot(self._x)

        # G - 2x1 matrix representing the control input (models affect of acceleration on position and velocity)
        # acceleration is assumed to be a random variable with a variance of accel_variance
        G = np.array([0.5 * dt**2, dt]).reshape((2, 1))
        # update covariance matrix (representing uncertainty in the state)
        new_P = F.dot(self._P).dot(F.T) + G.dot(G.T) * self._accel_variance

        self._P = new_P
        self._x = new_x

    @property
    def covariance(self) -> np.ndarray:
        """The covariance of the current state vector"""
        return self._P

    @property
    def mean(self) -> np.ndarray:
        """The mean of the current state vector"""
        return self._x

    @property
    def pos(self) -> float:
        """The current position"""
        return self._x[0]

    @property
    def vel(self) -> float:
        """The current velocity"""
        return self._x[1]
