from unittest import TestCase
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
