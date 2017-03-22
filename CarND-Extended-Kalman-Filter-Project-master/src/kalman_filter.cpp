#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {
}

KalmanFilter::~KalmanFilter() {
}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
		MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
	x_ = x_in;
	P_ = P_in;
	F_ = F_in;
	H_ = H_in;
	R_ = R_in;
	Q_ = Q_in;
}

void KalmanFilter::Predict() {
	// Kalman and Extended Kalman Filter Predict step
	VectorXd process_noise = VectorXd(4, 1);
	process_noise << x_[2], x_[3], x_[2], x_[3];
	x_ = F_ * x_; //+ process_noise;
	P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
	// Kalman Filter Update step
	VectorXd y = z - H_ * x_;
	MatrixXd HT = H_.transpose();
	MatrixXd S = H_ * P_ * HT + R_;
	MatrixXd K = P_ * HT * S.inverse();
	MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());

	// Update new state based on Kalman Gain
	x_ = x_ + (K * y);
	P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
	// Extended Kalman Filter Update step

}

VectorXd KalmanFilter::CartesianToPolar(const VectorXd &x_state) {
	VectorXd polar_state = VectorXd(3);
	double px = x_state[0];
	double py = x_state[1];
	double vx = x_state[2];
	double vy = x_state[3];

	double ro = sqrt(px * px + py * py);
//	double phi = atan(py / px);
	double phi = atan2(py, px);
	// Normalize phi
//	if (-M_PI > phi) {
//		phi += 2 * M_PI;
//	} else if (M_PI < phi) {
//		phi -= 2 * M_PI;
//	}

	if (-M_PI > phi || M_PI < phi) {
		std::cerr << "Normalization problem.." << std::endl;
	}

	std::cout << "Angle Phi: " << phi << std::endl;
	double denom = px * px + py * py;
	double ro_dot = 0.0;
	if (denom > 0.000001) {
		ro_dot = (px * vx + py * vy) / sqrt(denom);
	}
	polar_state[0] = ro;
	polar_state[1] = phi;
	polar_state[2] = ro_dot;

	return polar_state;
}
