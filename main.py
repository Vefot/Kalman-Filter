import numpy as np
import matplotlib.pyplot as plt
from kf import KF

plt.ion()
plt.figure()

kf = KF(initial_x=0.0, initial_v=1.0, accel_variance=0.1)

DT = 0.1
NUM_STEPS = 1000

mus = []  # mean of the state
covs = []  # covariance

for _ in range(NUM_STEPS):
    mus.append(kf.mean)
    covs.append(kf.covariance)
    kf.predict(dt=DT)

plt.subplot(2, 1, 1)
plt.title("Position")
# plot the mean of the state
plt.plot([mu[0] for mu in mus], "r")
# plot the uncertainty in the state - see a cone of uncertainty widening over time
plt.plot([mu[0] - 2 * np.sqrt(covs[0, 0]) for mu, covs in zip(mus, covs)], "r--")
plt.plot([mu[0] + 2 * np.sqrt(covs[0, 0]) for mu, covs in zip(mus, covs)], "r--")

plt.subplot(2, 1, 2)
plt.title("Velocity")
plt.plot([mu[1] for mu in mus], "r")
plt.plot([mu[1] - 2 * np.sqrt(covs[1, 1]) for mu, covs in zip(mus, covs)], "r--")
plt.plot([mu[1] + 2 * np.sqrt(covs[1, 1]) for mu, covs in zip(mus, covs)], "r--")


plt.show()
plt.ginput(1, timeout=0)
