#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {
}

Tools::~Tools() {
}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
	const vector<VectorXd> &ground_truth) {
	// RMSE calculation
	int size = estimations.size();
	VectorXd rmse(4);
	rmse << 0, 0, 0, 0;

	double err = 0.0;
	for (int i = 0; i < size; i++) {
		VectorXd diff = (estimations[i] - ground_truth[i]);
		rmse += (diff.array() * diff.array()).matrix();
	}

	rmse = (rmse.array() / size).sqrt();

	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
	// Calculating Jakobian Matrix
	// Extract values from state matrix
	double px = x_state(0);
	double py = x_state(1);
	double vx = x_state(2);
	double vy = x_state(3);

	// Calculate denominator variations
	double denom = px * px + py * py;
	double sdenom = sqrt(denom);
	double cdenom = pow(denom, 1.5);

	// Jakobian matrix initialization
	MatrixXd Hj(3, 4);
	Hj << 0.0, 0.0, 0.0, 0.0,
		  0.0, 0.0, 0.0, 0.0,
		  0.0, 0.0, 0.0, 0.0;

	// Chech for division by zero
	if (denom > 0.000001) {
		// If denominator is not zero, calculate jakobian matrix
		Hj << px / sdenom, py / sdenom, 0, 0, -py / denom, px / denom, 0, 0, (py
				* (vx * py - vy * px)) / cdenom, (px * (vy * px - vx * py))
				/ cdenom, px / sdenom, py / sdenom;
	} else {
		std::cout << "Jakobian is bad.." << std::endl;
		std::cout << "Jakobian Matrix: " << Hj << std::endl;
	}


	return Hj;
}
