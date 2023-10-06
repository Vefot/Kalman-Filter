from unittest import TestCase
import numpy as np
from kf import KF


class TestKF(TestCase):
    def test_can_construct_with_x_and_v(self):
        """
        Tests that the KF class can be constructed with initial x and v
        and that the state vector is initialized with the correct values (x and v)
        """
        x = 0.2
        v = 2.3
        kf = KF(initial_x=x, initial_v=v, accel_variance=1.2)

        # test that the first element of the state vector is the initial x
        self.assertAlmostEqual(kf.pos, x)

        # test that the second element of the state vector is the initial v
        self.assertAlmostEqual(kf.vel, v)

    def test_after_calling_predict_x_and_p_are_right_shape(self):
        x = 0.2
        v = 2.3
        kf = KF(initial_x=x, initial_v=v, accel_variance=1.2)
        kf.predict(dt=0.1)

        # the covariance matrix should be 2x2
        self.assertEqual(kf.covariance.shape, (2, 2))

        # the state vector should be 2x1
        self.assertEqual(kf.mean.shape, (2,))

    def test_calling_predict_increases_state_uncertainty(self):
        """
        Uncertainty in the state should increase with time,
        so P should be larger after calling predict
        """
        x = 0.2
        v = 2.3
        kf = KF(initial_x=x, initial_v=v, accel_variance=1.2)

        for i in range(10):
            # det is the determinant of the covariance matrix (a measure of uncertainty in the state)
            det_before = np.linalg.det(kf.covariance)
            kf.predict(dt=0.1)
            det_after = np.linalg.det(kf.covariance)

            self.assertGreater(det_after, det_before)
            print(det_before, det_after)

    def test_calling_update_decreases_state_uncertainty(self):
        """
        Calling update should decrease uncertainty in the state
        because we are getting new information about the state
        from e.g. a sensor measurement
        """
        x = 0.2
        v = 2.3

        kf = KF(initial_x=x, initial_v=v, accel_variance=1.2)

        det_before = np.linalg.det(kf.covariance)
        kf.update(measurement_value=0.1, measurement_variance=0.01)
        det_after = np.linalg.det(kf.covariance)

        self.assertLess(det_after, det_before)
