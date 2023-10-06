package main

import "fmt"

type KalmanFilter struct {
	xEstimate float64 // Estimated state (position)
	xError    float64 // Estimated error in state
}

func NewKalmanFilter(initialEstimate, initialError float64) *KalmanFilter {
	return &KalmanFilter{
		xEstimate: initialEstimate,
		xError:    initialError,
	}
}

func (kf *KalmanFilter) Predict(acceleration, dt, processNoise float64) {
	// Prediction step
	// Update the state estimate and error estimate based on motion model
	// Assuming a constant velocity model
	// x = x + (velocity * dt)
	// velocity remains the same

	// Predicted state estimate
	xPredicted := kf.xEstimate + (kf.xError * dt)

	// Predicted error estimate
	xErrorPredicted := kf.xError + processNoise

	kf.xEstimate = xPredicted
	kf.xError = xErrorPredicted
}

func (kf *KalmanFilter) Update(measurement, measurementNoise float64) {
	// Update step
	// Update the state estimate and error estimate based on measurement

	// Calculate Kalman gain
	kalmanGain := kf.xError / (kf.xError + measurementNoise)

	// Update the state estimate
	kf.xEstimate = kf.xEstimate + kalmanGain*(measurement-kf.xEstimate)

	// Update the error estimate
	kf.xError = (1 - kalmanGain) * kf.xError
}

func main() {
	// Initial state estimate and error estimate
	initialEstimate := 0.0
	initialError := 1.0

	// Process noise and measurement noise
	processNoise := 0.001
	measurementNoise := 0.1

	// Create a Kalman Filter
	kf := NewKalmanFilter(initialEstimate, initialError)

	// Simulated measurements
	measurements := []float64{1.1, 1.9, 3.0, 3.9, 5.0}

	// Time step
	dt := 1.0

	// Perform prediction and update steps for each measurement
	for _, measurement := range measurements {
		kf.Predict(0, dt, processNoise)
		kf.Update(measurement, measurementNoise)
		fmt.Printf("Estimated Position: %.2f\n", kf.xEstimate)
	}
}
