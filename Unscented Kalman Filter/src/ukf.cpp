#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#include <math.h>

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

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.8;

  // // For automatic tuning of std_a_ and std_yawdd_
  // ifstream settings_file("settings.txt", ifstream::in);
  // settings_file>>std_a_>>std_yawdd_;


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

  is_initialized_ = false;

  n_x_ = 5;
  n_aug_ = n_x_ + 2;
  lambda_ = 3 - n_aug_;

  weights_ = VectorXd::Zero(2*n_aug_+1);
  weights_ << lambda_/(lambda_+n_aug_), VectorXd::Constant(2*n_aug_, 0.5/(n_aug_+lambda_));


  Xsig_pred_ = MatrixXd::Zero(n_x_, 2*n_aug_+1);

  R_laser = MatrixXd(2,2);
  R_laser <<  std_laspx_*std_laspx_, 0,
              0, std_laspy_*std_laspy_;
  
  //add measurement noise covariance matrix
  R_radar = MatrixXd(3,3);
  R_radar <<  std_radr_*std_radr_, 0, 0,
              0, std_radphi_*std_radphi_, 0,
              0, 0,std_radrd_*std_radrd_;


}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if(!is_initialized_) {
    cout<<"UKF: "<<endl;

    if(meas_package.sensor_type_ == MeasurementPackage::LASER){
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
    }
    else if(meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      double rho = meas_package.raw_measurements_[0];
      double phi = meas_package.raw_measurements_[1];
      x_ << rho*cos(phi), rho*sin(phi), 0, 0, 0;
    }

    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;

    return;
  }

  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0; //dt - expressed in seconds
  time_us_ = meas_package.timestamp_;

  Prediction(dt);

  if(meas_package.sensor_type_ == MeasurementPackage::LASER){
    UpdateLidar(meas_package);
  }
  else if(meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  }

}

MatrixXd UKF::GenerateAugSigmaPoints(){
  int n_pts = 2*n_aug_ + 1;

  VectorXd x_aug = VectorXd::Zero(n_aug_);
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
  MatrixXd Xsig_aug = MatrixXd::Zero(n_aug_, n_pts);

  x_aug.head(n_x_) = x_;
  
  P_aug.topLeftCorner(n_x_,n_x_) = P_;
  P_aug(n_x_,n_x_) = std_a_*std_a_;
  P_aug(n_x_+1,n_x_+1) = std_yawdd_*std_yawdd_;
  
  MatrixXd A = P_aug.llt().matrixL();

  double scale = sqrt(lambda_ + n_aug_);
  
  Xsig_aug << x_aug, (scale*A).colwise() + x_aug, (-scale*A).colwise() + x_aug;
  return Xsig_aug;
}

void UKF::PredictSigmaPoints(const MatrixXd &Xsig_aug, double delta_t){
  for(int i = 0; i< 2*n_aug_+1; i++) {
    //extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin(yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
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
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
}

void AngleNormalization(MatrixXd &a, int i){
  double angle;
  for(int j=0, c=a.cols(); j<c; ++j){
    angle = a(i,j);
    angle -= 2*M_PI*floor(angle/(2*M_PI));
    while (angle > M_PI) angle-=2.*M_PI;
    while (angle < -M_PI) angle+=2.*M_PI;
    a(i,j) = angle;
  }
}

void UKF::PredictMeanCovariance(){
  //predicted state mean
  x_ = Xsig_pred_*weights_;

  //predicted state covariance matrix
  MatrixXd x_diff = Xsig_pred_.colwise() - x_;
  AngleNormalization(x_diff, 3);

  P_ = x_diff*weights_.asDiagonal()*x_diff.transpose();
}


/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  // cout<<"Prediction"<<endl;
  MatrixXd Xsig_aug = GenerateAugSigmaPoints();
  PredictSigmaPoints(Xsig_aug, delta_t);
  PredictMeanCovariance();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  // cout<<"UpdateLidar"<<endl;
  int n_z = 2; // lidar measurement dimentionality
  
  MatrixXd Zsig = Xsig_pred_.topRows(n_z);

  NIS_laser_ = PredictMeasurement(meas_package.raw_measurements_, Zsig, R_laser, -1, -1);
  cout<<"NIS_laser_ = "<<NIS_laser_<<endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  // cout<<"UpdateRadar"<<endl;
  int n_pts = 2 * n_aug_ + 1; // number of sigma points
  int n_z = 3; // radar measurement dimentionality

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, n_pts);

  //transform sigma points into measurement space
  for (int i = 0; i < n_pts; i++) {  //2n+1 simga points

    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v   = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double vx = cos(yaw)*v;
    double vy = sin(yaw)*v;

    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Zsig(1,i) = atan2(p_y,p_x);                                 //phi
    Zsig(2,i) = (p_x*vx + p_y*vy) / Zsig(0,i);                  //r_dot
  }

  NIS_radar_ = PredictMeasurement(meas_package.raw_measurements_, Zsig, R_radar, 1, 3);
  cout<<"NIS_radar_ = "<<NIS_radar_<<endl;
}

double UKF::PredictMeasurement(const VectorXd &z, const MatrixXd &Zsig, const MatrixXd &R, int zangle_idx, int xangle_idx){
  //mean predicted measurement
  VectorXd z_pred = Zsig*weights_;
  MatrixXd zsig_diff = Zsig.colwise() - z_pred;
  MatrixXd x_diff = Xsig_pred_.colwise() - x_;

  if(zangle_idx >= 0) AngleNormalization(zsig_diff, zangle_idx);
  if(xangle_idx >= 0) AngleNormalization(x_diff, xangle_idx);

  MatrixXd S = zsig_diff*weights_.asDiagonal()*zsig_diff.transpose() + R;
  MatrixXd Sinv = S.inverse();

  //create matrix for cross correlation Tc
  MatrixXd Tc = x_diff*weights_.asDiagonal()*zsig_diff.transpose();

  //Kalman gain K;
  MatrixXd K = Tc * Sinv;

  //residual
  VectorXd z_diff = z - z_pred;

  //angle normalization
  if(zangle_idx >= 0){
    double angle = z_diff(zangle_idx);
    while (angle>M_PI) angle-=2.*M_PI;
    while (angle<-M_PI) angle+=2.*M_PI;
    z_diff(zangle_idx) = angle;
  }

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();

  double NIS = (z_diff.transpose()*Sinv*z_diff)(0);
  return NIS;
}