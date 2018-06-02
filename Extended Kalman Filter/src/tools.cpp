#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::cout;
using std::endl;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
  const vector<VectorXd> &ground_truth) {

	VectorXd rmse = VectorXd::Zero(4);

	// check the validity of the following inputs:
	//  * the estimation vector size should not be zero
	//  * the estimation vector size should equal ground truth vector size
	if(estimations.size() != ground_truth.size() || estimations.size() == 0){
		cout << "Invalid estimation or ground_truth data" << endl;
    return rmse;
  }

  	//accumulate squared residuals
  for(unsigned int i=0; i < estimations.size(); ++i){
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array()*residual.array();
    rmse += residual;
  }

  	//calculate the mean
  rmse = (rmse/estimations.size()).array().sqrt();
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj(3,4);
  
  //recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  //pre-compute a set of terms to avoid repeated calculation
  float c2 = px*px+py*py;
  float c1 = sqrt(c2);
  float c3 = c1*c2;

  //check division by zero
  if(fabs(c1) < 1e-6){
    cout << "CalculateJacobian () - Error - Division by Zero" << endl;
    return Hj;
  }

  //compute the Jacobian matrix
  Hj << (px/c1), (py/c1), 0, 0,
        -(py/c2), (px/c2), 0, 0,
        py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c1, py/c1;

  return Hj;
}