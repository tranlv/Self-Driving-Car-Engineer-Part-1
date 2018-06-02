#include "kalman_filter.h"
#include "tools.h"
#include <iostream>

#define PI 3.141592653589793238463

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

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
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;

}

void KalmanFilter::UpdateCommon(const VectorXd &y) {
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}


void KalmanFilter::Update(const VectorXd &z) {
  UpdateCommon(z - H_*x_);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  float px = x_(0), py = x_(1), vx = x_(2), vy = x_(3);
  float rho = sqrt(px*px + py*py), phi = atan2(py, px);

  if(rho < 1e-6){
    cout << "UpdateEKF () - Error - Division by Zero" << endl;
    return;
  }

  VectorXd z_pred(3);
  z_pred << rho, phi, (px*vx + py*vy)/rho;

  VectorXd y = z - z_pred;

  // make sure that angle is between -PI and PI
  while(y(1) > PI) y(1) -= 2*PI;
  while(y(1) < -PI) y(1) += 2*PI;

  UpdateCommon(y);
}