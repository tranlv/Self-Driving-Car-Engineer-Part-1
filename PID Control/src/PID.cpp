#include "PID.h"
#include <cmath>
#include <iostream>
#include <limits>
using namespace std;

/*
* TODO: Complete the PID class.
*/



PID::PID(double kp, double ki, double kd): Kp(kp), Ki(ki), Kd(kd) {
  p_error = 0.0;
  i_error = 0.0;
  d_error = 0.0;
  last_cte = 0.0;
  dKp = kp/4;
  dKi = ki/4;
  dKd = Kd/4;
  sse = 0;
  count = 0;
  cur_param = 0;
  cur_twiddle_dir = true;
  best_mse = 1e100;
  first_time = true;
}

PID::~PID() {}


void PID::UpdateError(double cte) {

	d_error = cte - p_error; 
	p_error = cte;
  i_error += cte;
  last_cte = cte;
  count++;
  sse += cte*cte;
}

double PID::TotalError() {
	return -Kp * p_error - Kd * d_error - Ki * i_error;
}

void PID::reset_twiddle(){
  dKp = Kp/10, dKi = Ki/10, dKd = Kd/10;  
  sse = 0, count = 0;
  cur_param = 0, cur_twiddle_dir = true;
  best_mse = 1e100;
  first_time = true;
}

double& PID::get_param_ref(int i){
  if(i==0) return Kp;
  if(i==1) return Ki;
  return Kd;
}

double& PID::get_dparam_ref(int i){
  if(i==0) return dKp;
  if(i==1) return dKi;
  return dKd;
}

void PID::twiddle(){
  double mse = sse/count;
  sse = 0;
  count = 0;

  if(first_time){
    best_mse = mse;
    Kp += dKp;
    first_time = false;
    return;
  }

  double& param = get_param_ref(cur_param);
  double& dparam = get_dparam_ref(cur_param);

  if(mse < best_mse){
    best_mse = mse;
    dparam *= 1.1;
    cout<<"*** NEW BEST PARAMS"<<endl;
  }
  else {
    if(cur_twiddle_dir) {
      cur_twiddle_dir = false;
      param -= 2*dparam;
      return;
    }
    else {
      param += dparam;
      dparam *= 0.9;
    }
  }

  cur_param =  (cur_param + 1)%3;
  cur_twiddle_dir = true;
  get_param_ref(cur_param) += get_dparam_ref(cur_param);      

  cout<<"--------------------------------"<<endl;
}

double PID::get_twiddle_error() const {
  return dKp/Kp + dKd/Kd + dKi/Ki;
}