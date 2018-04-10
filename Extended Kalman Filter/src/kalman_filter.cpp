#include "kalman_filter.h"



using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

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
  /**
  TODO:
    * predict the state
  */
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ *  Ft +  Q_; 
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
  MatrixXd y_ =  z - H_ * x_;
  MatrixXd Ht_ = H_.transpose();
  MatrixXd S_ = H_* P_ * Ht_ + R_;
  MatrixXd Si_ = S_.inverse();
  MatrixXd K_ = P_ * Ht_ * Si_;
  x_ = x_ + (K_ * y_);
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K_ * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */
  float px = z(0);
  float py = z(1);
  float vx = z(2);
  float vy = z(3);
  float temp1 = pow(px,2) + pow(py,2);

  if(fabs(temp1) < 0.0001){
    cout << "UpdateEKF () - Error - Division by Zero" << endl;
  }

  float ro = sqrt(temp1)
  float theta = atan(px/py);
  float ro_dot = (px*vx + py *vy)/(sqrt(temp1));
  VectorXd h = VectorXd(3)
  h << ro,theta, ro_dot;
  MatrixXd y_ =  z - h;

  MatrixXd Ht_ = H_.transpose();
  MatrixXd S_ = H_* P_ * Ht_ + R_;
  MatrixXd Si_ = S_.inverse();
  MatrixXd K_ = P_ * Ht_ * Si_;
  x_ = x_ + (K_ * y_);
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K_ * H_) * P_;
}
