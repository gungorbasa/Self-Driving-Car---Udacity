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
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	// Calculate denominator variations
	double denom = px * px + py * py;
	double sdenom = sqrt(denom);
	double cdenom = cbrt(denom);

	// Jakobian matrix initialization
	MatrixXd Hj(3, 4);
	// Chech for division by zero
	if (fabs(denom) > 0.000001) {
		// If denominator is not zero, calculate jakobian matrix
		Hj << px / sdenom, py / sdenom, 0, 0, -py / denom, px / denom, 0, 0, (py
				* (vx * py - vy * px)) / cdenom, (px * (vy * px - vx * py))
				/ cdenom, px / sdenom, py / sdenom;
	} else {
		std::cout << "Jakobian is bad.." << std::endl;
	}

	return Hj;
}
