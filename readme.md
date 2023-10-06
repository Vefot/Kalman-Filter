# Python Kalman Filter Implementation

Goal: Estimate X, Y position of a moving object using a Kalman Filter

## Overview

There are two stages of the Kalman Filter: Prediction and Measurement/Correction

Estimate the speed of an object, given some measurements.

These are: 

z = x (location) + Er1, Er~N(O,T2^2)

Only measure the (noisy) location of X we can get an estimate of the location AND the speed. The KF will give us estimate of the location and speed plus uncertainty estimate.

**State Vector** 

X (Xk) is our state vector of the location and speed [Xk Vk]

We also have our measurement: Zh = [Zh] (location)

**Define how our system is evolving through time** - Time evolution

Xk + 1 = Xk + Vk * dt + acceleration - The location is the location plus speed times time (dt=delta time) plus acceleration

Vk + 1 = Vk + acceleration - The speed is the speed plus acceleration

Vector form ("F" and "G" vectors):

Xk = [1 dt 0 0] * [Xk Vk Xk Vk] + [0 0 dt 0] * [acceleration]


Speed can change due to noise, it is hard to measure. Acceleration is the noise.

How to incorporate measurement information into the state vector?

Zk = Xk + Er - Z (representing the measurement) is the location plus some error (Er)

Vector form (H vector): 

Zk = [1 0] * [Xk Vk] + Er

## Kalman Filter Formulae

### Prediction step

Propogate state Xk -> Xk + 1

Xk ~ N(Xk, Pk) - Xk is the mean, Pk is the covariance matrix

The mean = Xk + 1 = F * Xk

Pk + 1 = F * Pk * F^T + Qk - Qk is the covariance matrix of the noise

### Measurement/correction step
Incorporate the knowledge of Zk into the state vector (Xk)

**The Update Equation**

Y = Zk - H * Xk - Y is the residual (difference between the measurement and the prediction)

Sk = H * Pk * H^T + Rk - Rk is the covariance matrix of the noise, Pk is the covariance matrix of the state vector

Kk = Pk * H^T * Sk^-1 - Kk is the Kalman gain

Xk + 1 = Xk + Kk * Y - Xk is the mean, Kk is the Kalman gain, Y is the residual

Pk + 1 = (I - Kk * H) * Pk - Pk is the covariance matrix of the state vector, I is the identity matrix