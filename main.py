import numpy as np
import matplotlib.pyplot as plt
from kf import KF

plt.ion()
plt.figure()

real_x = 0.0
measurement_variance = 0.1**2  # simulate noise in the measurement
real_v = 0.5

kf = KF(initial_x=0.0, initial_v=1.0, accel_variance=0.1)

DT = 0.1
NUM_STEPS = 1000
MEASUREMENT_EVERY_STEPS = 20


mus = []  # mean of the state
covs = []  # covariance
real_xs = []
real_vs = []

for step in range(NUM_STEPS):
    # change the speed after step 500..
    if step > 500:
        real_v *= 0.9

    mus.append(kf.mean)
    covs.append(kf.covariance)

    # update the position - simulate in this manner:
    real_x = real_x + DT * real_v

    kf.predict(dt=DT)

    # when we get a sensor reading, update the kalman filter
    # instead of giving the kf the real x we give it a slightly
    # noisy version
    if step != 0 and step % MEASUREMENT_EVERY_STEPS == 0:
        kf.update(
            measurement_value=real_x
            + np.random.randn() * np.sqrt(measurement_variance),
            measurement_variance=measurement_variance,
        )

    real_xs.append(real_x)
    real_vs.append(real_v)


plt.subplot(2, 1, 1)
plt.title("Position")
# plot the mean of the state
plt.plot([mu[0] for mu in mus], "r")
plt.plot(real_xs, "b")


# plot the uncertainty in the state - see a cone of uncertainty widening over time
plt.plot([mu[0] - 2 * np.sqrt(covs[0, 0]) for mu, covs in zip(mus, covs)], "r--")
plt.plot([mu[0] + 2 * np.sqrt(covs[0, 0]) for mu, covs in zip(mus, covs)], "r--")

plt.subplot(2, 1, 2)
plt.title("Velocity")
plt.plot([mu[1] for mu in mus], "r")
plt.plot(real_vs, "b")

plt.plot([mu[1] - 2 * np.sqrt(covs[1, 1]) for mu, covs in zip(mus, covs)], "r--")
plt.plot([mu[1] + 2 * np.sqrt(covs[1, 1]) for mu, covs in zip(mus, covs)], "r--")


plt.show()
plt.ginput(1)
