#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
	is_initialized_ = false;

	previous_timestamp_ = 0;

	// initializing matrices
	R_laser_ = MatrixXd(2, 2);
	R_radar_ = MatrixXd(3, 3);
	H_laser_ = MatrixXd(2, 4);
	Hj_ = MatrixXd(3, 4);

	//measurement covariance matrix - laser
	R_laser_ << 0.0225, 0, 0, 0.0225;

	//measurement covariance matrix - radar
	R_radar_ << 0.09, 0, 0, 0, 0.0009, 0, 0, 0, 0.09;

	/**
	 TODO:
	 * Finish initializing the FusionEKF.
	 * Set the process and measurement noises
	 */
	Q_ = MatrixXd(4, 4);
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {
}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

	/*****************************************************************************
	 *  Initialization
	 ****************************************************************************/
	if (!is_initialized_) {
		/**
		 TODO:
		 * Initialize the state ekf_.x_ with the first measurement.
		 * Create the covariance matrix.
		 * Remember: you'll need to convert radar from polar to cartesian coordinates.
		 */
		// first measurement
		cout << "EKF: " << endl;
		ekf_.x_ = VectorXd(4);
		ekf_.x_ << 1, 1, 1, 1;

		if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
			float phi = measurement_pack.raw_measurements_[1];
			float x = measurement_pack.raw_measurements_[0] * cos(phi);
			float y = measurement_pack.raw_measurements_[0] * sin(phi);
			float vx = 0.0;
			float vy = 0.0;

			VectorXd cartesian_state = VectorXd(4);
			cartesian_state << x, y, vx, vy;
			ekf_.x_ = cartesian_state;
		} else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
			ekf_.x_ = measurement_pack.raw_measurements_;
		}

		// done initializing, no need to predict or update
		is_initialized_ = true;
		return;
	}

	/*****************************************************************************
	 *  Prediction
	 ****************************************************************************/

	/**
	 TODO:
	 * Update the state transition matrix F according to the new elapsed time.
	 - Time is measured in seconds.
	 * Update the process noise covariance matrix.
	 * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
	 */
	float noise_ax = 9;
	float noise_ay = 9;
	float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
	float dt2 = dt * dt;
	float dt3 = dt * dt2;
	float dt4 = dt * dt3;
	previous_timestamp_ = measurement_pack.timestamp_;

	// Calculating Process Covariance Matrix
	Q_[0] = dt4/4*noise_ax; Q_[2] = dt3/2*noise_ax; Q_[5] = dt4/4*noise_ay;
	Q_[7] = dt3/2*noise_ay; Q_[8] = dt3/2*noise_ax; Q_[10] = dt2*noise_ax;
	Q_[13] = dt3/2*noise_ay; Q_[15] = dt2*noise_ay;

	ekf_.Q_ = Q_;

	// Calculating F matrix
	ekf_.F_ = MatrixXd(4,4);
	ekf_.F_ << 1, 0, dt, 0,
			   0, 1, 0, dt,
			   0, 0, 1, 0,
			   0, 0, 0, 1;


	ekf_.Predict();

	/*****************************************************************************
	 *  Update
	 ****************************************************************************/

	/**
	 TODO:
	 * Use the sensor type to perform the update step.
	 * Update the state and covariance matrices.
	 */

	if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
		// Radar updates
		ekf_.R_ = R_radar_;
	} else {
		// Laser updates
		ekf_.R_ = R_laser_;
	}

	// print the output
	cout << "x_ = " << ekf_.x_ << endl;
	cout << "P_ = " << ekf_.P_ << endl;
}