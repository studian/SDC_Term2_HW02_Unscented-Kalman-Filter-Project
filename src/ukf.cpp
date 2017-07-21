#include "ukf.h"
//#include "tools.h" // add
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
* Initializes Unscented Kalman filter
*/
UKF::UKF() {
	// if this is false, laser measurements will be ignored (except during init)
	use_laser_ = true;

	// if this is false, radar measurements will be ignored (except during init)
	use_radar_ = true;

	// initial state vector
	x_ = VectorXd(5);

	// initial covariance matrix
	P_ = MatrixXd(5, 5);

	P_ = MatrixXd::Identity(5, 5); // add

	// Process noise standard deviation longitudinal acceleration in m/s^2
	std_a_ = 2; // add 30->2

	// Process noise standard deviation yaw acceleration in rad/s^2
	std_yawdd_ = 0.3; // add 30->0.3

	// Laser measurement noise standard deviation position1 in m
	std_laspx_ = 0.15;

	// Laser measurement noise standard deviation position2 in m
	std_laspy_ = 0.15;

	// Radar measurement noise standard deviation radius in m
	std_radr_ = 0.3;

	// Radar measurement noise standard deviation angle in rad
	std_radphi_ = 0.03;

	// Radar measurement noise standard deviation radius change in m/s
	std_radrd_ = 0.3;

        // add start
	// State dimension
	n_x_ = 5;

	// Augmented state dimension
	n_aug_ = 7;

	//create augment sigma point matrix
	Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);

	// predicted sigma points matrix
	Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

	// Sigma point spreading parameter
	lambda_ = 3 - n_aug_;

	// Weights of sigma points
	weights_ = VectorXd(2 * n_aug_ + 1);
	weights_.segment(1, 2 * n_aug_).fill(0.5 / (n_aug_ + lambda_));
	weights_(0) = lambda_ / (lambda_ + n_aug_);

	// Sensor's measurement size
	n_radar_ = 3; // rho, phi, dot_rho
	n_lidar_ = 2; // px, py

				  // Measurement covariance matrices
	R_lidar_ = MatrixXd(n_lidar_, n_lidar_);
	R_radar_ = MatrixXd(n_radar_, n_radar_);

	// Initialize Normalized Innovation Squared (NIS) value for both sensors
	NIS_laser_ = 0.;
	NIS_radar_ = 0.;
        // add end
}

UKF::~UKF() {}

/**
* @param {MeasurementPackage} meas_package The latest measurement data of
* either radar or laser.
*/
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
        // add start

	if (!is_initialized_) {

		if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {

			float rho = meas_package.raw_measurements_(0);
			float phi = meas_package.raw_measurements_(1);
			float dot_rho = meas_package.raw_measurements_(2);

			// Convert from polar to cartesian coordinates
			float px = rho * cos(phi);
			float py = rho * sin(phi);

			// Initialize state
			x_ << px, py, dot_rho, 0.0, 0.0;
		}
		else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {

			// Extract values from measurement
			float px = meas_package.raw_measurements_(0);
			float py = meas_package.raw_measurements_(1);

			// Initialize state
			x_ << px, py, 0.0, 0.0, 0.0;
		}

		// Update last measurement
		time_us_ = meas_package.timestamp_;

		is_initialized_ = true;
		return;

	}

	// Prediction

	// Compute elapsed time from last measurement
	float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;

	// Update last measurement
	time_us_ = meas_package.timestamp_;

	Prediction(dt);

	// Update

	if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
		// Radar updates 
		UpdateRadar(meas_package);
	else
		// Laser updates
		UpdateLidar(meas_package);
        // add end
}

void UKF::Prediction(double delta_t) {
        // add start

	//.. Augmented Sigma Points
	//create augmented mean vector
	VectorXd x_aug = VectorXd(n_aug_);

	//create augmented state covariance
	MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

	//create augmented mean state
	x_aug.head(5) = x_;
	x_aug(5) = 0;
	x_aug(6) = 0;

	//create augmented covariance matrix
	P_aug.fill(0.0);
	P_aug.topLeftCorner(5, 5) = P_;
	P_aug(5, 5) = std_a_*std_a_;
	P_aug(6, 6) = std_yawdd_*std_yawdd_;

	//create square root matrix
	MatrixXd L = P_aug.llt().matrixL();

	//create augmented sigma points
	Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);
	Xsig_aug_.col(0) = x_aug;
	for (int i = 0; i< n_aug_; i++)
	{
		Xsig_aug_.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
		Xsig_aug_.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
	}

        //.. Sigma Point Prediction in the delta_t;

	//predict sigma points
	for (int i = 0; i< (2 * n_aug_ + 1); i++)
	{
		//extract values for better readability
		double p_x = Xsig_aug_(0, i);
		double p_y = Xsig_aug_(1, i);
		double v = Xsig_aug_(2, i);
		double yaw = Xsig_aug_(3, i);
		double yawd = Xsig_aug_(4, i);
		double nu_a = Xsig_aug_(5, i);
		double nu_yawdd = Xsig_aug_(6, i);

		if (fabs(p_x) < 0.001 && fabs(p_y) < 0.001) {
			p_x = 0.1;
			p_y = 0.1;
		}

		//predicted state values
		double px_p, py_p;

		//avoid division by zero
		if (fabs(yawd) > 0.001) {
			px_p = p_x + v / yawd * (sin(yaw + yawd*delta_t) - sin(yaw));
			py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd*delta_t));
		}
		else {
			px_p = p_x + v*delta_t*cos(yaw);
			py_p = p_y + v*delta_t*sin(yaw);
		}

		double v_p = v;
		double yaw_p = yaw + yawd*delta_t;
		double yawd_p = yawd;

		//add noise
		px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
		py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
		v_p = v_p + nu_a*delta_t;

		yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
		yawd_p = yawd_p + nu_yawdd*delta_t;

		//write predicted sigma point into right column
		Xsig_pred_(0, i) = px_p;
		Xsig_pred_(1, i) = py_p;
		Xsig_pred_(2, i) = v_p;
		Xsig_pred_(3, i) = yaw_p;
		Xsig_pred_(4, i) = yawd_p;
	}

	//predicted state mean
	x_.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
		x_ = x_ + weights_(i) * Xsig_pred_.col(i);
	}

	//predicted state covariance matrix
	P_.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

												// state difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;

		// normalize
		normalize_PI(x_diff(3));

		P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
	}
        // add end
}

/**
* Updates the state and the state covariance matrix using a laser measurement.
* @param {MeasurementPackage} meas_package
*/
void UKF::UpdateLidar(MeasurementPackage meas_package) {
// add start

	// Prediction
	// Project sigma points onto measurement space
	MatrixXd Zsig = MatrixXd(n_lidar_, 2 * n_aug_ + 1);

	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

		// extract values for better readibility
		double p_x = Xsig_pred_(0, i);
		double p_y = Xsig_pred_(1, i);

		// measurement model
		Zsig(0, i) = p_x;
		Zsig(1, i) = p_y;
	}

	//mean predicted measurement
	VectorXd z_pred = VectorXd(n_lidar_);
	z_pred.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		z_pred = z_pred + weights_(i) * Zsig.col(i);
	}

	// Predicted measurement covariance matrix
	MatrixXd S = MatrixXd(n_lidar_, n_lidar_);
	S.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {

		VectorXd z_diff = Zsig.col(i) - z_pred;

		// normalize
		normalize_PI(z_diff(1));

		S = S + weights_(i) * z_diff * z_diff.transpose();
	}

	// Handle measurement noise 
	R_lidar_ << std_laspx_ * std_laspx_, 0,
		0, std_laspy_ * std_laspy_;
	S = S + R_lidar_;

	// Update

	// Parse laser measurement
	VectorXd z = VectorXd(n_lidar_);
	z << meas_package.raw_measurements_[0],
		meas_package.raw_measurements_[1];

	// Ccompute cross correlation matrix
	MatrixXd Tc = MatrixXd(n_x_, n_lidar_);
	Tc.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {

		// Residual
		VectorXd z_diff = Zsig.col(i) - z_pred;

		// State difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;

		Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
	}

	// Compute Kalman gain;
	MatrixXd K = Tc * S.inverse();

	// Residual
	VectorXd z_diff = z - z_pred;

	// Update state mean and covariance matrix
	x_ = x_ + K * z_diff;
	P_ = P_ - K * S * K.transpose();

	// Compute NIS for laser sensor
	NIS_laser_ = (meas_package.raw_measurements_ - z_pred).transpose() * S.inverse() *
		(meas_package.raw_measurements_ - z_pred);
// add end
}

/**
* Updates the state and the state covariance matrix using a radar measurement.
* @param {MeasurementPackage} meas_package
*/
void UKF::UpdateRadar(MeasurementPackage meas_package) {
// add start

	// Prediction
	// Project sigma points onto measurement space
	MatrixXd Zsig = MatrixXd(n_radar_, 2 * n_aug_ + 1);

	for (int i = 0; i < 2 * n_aug_ + 1; i++) {

		// extract values for better readability
		double p_x = Xsig_pred_(0, i);
		double p_y = Xsig_pred_(1, i);
		double v = Xsig_pred_(2, i);
		double yaw = Xsig_pred_(3, i);

		double v1 = cos(yaw) * v;
		double v2 = sin(yaw) * v;

		// Measurement model
		Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);

		if (fabs(p_y) > 0.001 && fabs(p_x) > 0.001)
			Zsig(1, i) = atan2(p_y, p_x);
		else
			Zsig(1, i) = 0.0;

		if (fabs(sqrt(p_x * p_x + p_y * p_y)) > 0.001)
			Zsig(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y);
		else
			Zsig(2, i) = 0.0;
	}

	// Predicted measurement mean
	VectorXd z_pred = VectorXd(n_radar_);
	z_pred.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		z_pred = z_pred + weights_(i) * Zsig.col(i);
	}

	// Predicted measurement covariance matrix
	MatrixXd S = MatrixXd(n_radar_, n_radar_);
	S.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {

		VectorXd z_diff = Zsig.col(i) - z_pred;

		// nomalize
		normalize_PI(z_diff(1));

		S = S + weights_(i) * z_diff * z_diff.transpose();
	}

	// Handle measurement noise
	R_radar_ << std_radr_ * std_radr_, 0, 0,
		0, std_radphi_ * std_radphi_, 0,
		0, 0, std_radrd_ * std_radrd_;
	S = S + R_radar_;

	// Update
	// Parse radar measurement
	VectorXd z = VectorXd(n_radar_);
	z << meas_package.raw_measurements_[0],
		meas_package.raw_measurements_[1],
		meas_package.raw_measurements_[2];

	// Compute cross correlation matrix
	MatrixXd Tc = MatrixXd(n_x_, n_radar_);
	Tc.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  // iterate over sigma points

												// Residual
		VectorXd z_diff = Zsig.col(i) - z_pred;

		// nomalize
		normalize_PI(z_diff(1));

		// State difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;

		// nomalize
		normalize_PI(x_diff(3));

		Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
	}

	// Compute Kalman gain
	MatrixXd K = Tc * S.inverse();

	// Residual
	VectorXd z_diff = z - z_pred;

	// nomalize
	normalize_PI(z_diff(1));

	// Update state mean and covariance matrix
	x_ = x_ + K * z_diff;
	P_ = P_ - K * S * K.transpose();

	// Compute NIS for radar sensor
	NIS_radar_ = (meas_package.raw_measurements_ - z_pred).transpose() * S.inverse() *
		(meas_package.raw_measurements_ - z_pred);
// add end
}

// add start
void UKF::normalize_PI(double& phi)
{
	while (phi < -M_PI || phi > M_PI) {
		if (phi < -M_PI) phi += 2 * M_PI;
		else phi -= 2 * M_PI;
	}
}
// add end
