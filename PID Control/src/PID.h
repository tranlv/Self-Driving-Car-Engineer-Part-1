#ifndef PID_H
#define PID_H
#include <vector>
#include <ctime>

class PID {
public:
  /*
  * Errors
  */
  double p_error;
  double i_error;
  double d_error;
  double last_cte;

  double Kp, Ki, Kd;  

  double sse;
  int count;

  // twiddle parameters
  double dKp, dKi, dKd;

  // best mean squared errors so far
  double best_mse;


  // current parameter being twiddled
  // 0 is Kp, 1 is Ki and 2 is Kd
  int cur_param;

  // current direction of twiddle
  // true is positive twiddle and false if negative twiddle
  bool cur_twiddle_dir;

  // if current run is the first run (i.e. no twiddling yet)
  bool first_time;


  /*
  * Constructor
  */
  PID(double kp, double ki, double kd);

  /*
  * Destructor.
  */
  virtual ~PID();

  /*
  * Initialize PID.
  */
  void Init(double Kp, double Ki, double Kd);

  /*
  * Update the PID error variables given cross track error.
  */
  void UpdateError(double cte);

  /*
  * Calculate the total PID error.
  */
  double TotalError();

    void twiddle();

  void reset_twiddle();

  double get_twiddle_error() const;

  // get the reference of the parameter from the id
  double& get_param_ref(int param_id);

  // get the reference of the twiddle parameter from the id
  double& get_dparam_ref(int param_id);
};

#endif /* PID_H */
