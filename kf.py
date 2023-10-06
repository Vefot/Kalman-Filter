import numpy as np

# offsets of each variable in the state vector
iX = 0
iV = 1
NUMVARS = iV + 1


class KF:
    """
    One dimensional Kalman filter

    Instantiates with the following properties:

    initial_x: initial location
    initial_v: initial velocity
    accel_variance: variance in acceleration
    """

    def __init__(
        self, initial_x: float, initial_v: float, accel_variance: float
    ) -> None:
        # mean of state gaussian random variable
        self._x = np.zeros(NUMVARS)
        self._x[iX] = initial_x
        self._x[iV] = initial_v

        self._accel_variance = accel_variance

        # covariance matrix - represents uncertainty in the state
        self._P = np.eye(NUMVARS)  # eye returns a 2x2 identity matrix

    def predict(self, dt: float) -> None:
        """
        Predicts the state vector forward in time by dt (delta time) seconds
        """
        # x = F * x
        # p = F * p * F^T  * a

        # state transition matrix
        F = np.eye(NUMVARS)
        F[iX, iV] = dt

        # calculate the new state by multiplying state transition matrix with current x
        new_x = F.dot(self._x)

        # G - 2x1 matrix representing the control input (models affect of acceleration on position and velocity)
        # acceleration is assumed to be a random variable with a variance of accel_variance
        G = np.zeros((2, 1))
        G[iX] = 0.5 * dt**2
        G[iV] = dt
        # update covariance matrix (representing uncertainty in the state)
        new_P = F.dot(self._P).dot(F.T) + G.dot(G.T) * self._accel_variance

        self._P = new_P
        self._x = new_x

    def update(self, measurement_value: float, measurement_variance: float):
        # y = z - H x
        # S = H P Ht + R
        # K = P Ht S^-1
        # x = x + K y
        # P = (I - K H) * P

        H = np.zeros((1, NUMVARS))
        H[0, iX] = 1

        z = np.array([measurement_value])
        R = np.array([measurement_variance])

        y = z - H.dot(self._x)
        S = H.dot(self._P).dot(H.T) + R
        
        K = self._P.dot(H.T).dot(np.linalg.inv(S))

        new_x = self._x + K.dot(y)
        new_p = (np.eye(2) - K.dot(H)).dot(self._P)

        self._P = new_p
        self._x - new_x

    @property
    def covariance(self) -> np.array:
        """The covariance of the current state vector"""
        return self._P

    @property
    def mean(self) -> np.array:
        """The mean of the current state vector"""
        return self._x

    @property
    def pos(self) -> float:
        """The current position"""
        return self._x[iX]

    @property
    def vel(self) -> float:
        """The current velocity"""
        return self._x[iV]
