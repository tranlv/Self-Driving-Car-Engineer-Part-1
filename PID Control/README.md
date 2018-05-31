# **PID Control**


---

## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1(mac, linux), 3.81(Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `./install-mac.sh` or `./install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.
* Simulator. You can download these from the [project intro page](https://github.com/udacity/self-driving-car-sim/releases) in the classroom.

There's an experimental patch for windows in this [PR](https://github.com/udacity/CarND-PID-Control-Project/pull/3)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./pid`. 

Tips for setting up your environment can be found [here](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/23d376c7-0195-4276-bdf0-e02f1f3c665d)

## Reflection

1. The effect of the P, I, D component of the PID algorithm in their implementation

* P component increase the rate at which the car return to the track (the line of zero cte). However, it also tends to increases the fluctuations/swings of the car around the track.

* D component dampens the swings of the cars around the tracks. However it also tends to slows the rate at which the car corrects its cte. Thus, in curved road segments, along which the cars constantly need to adjusts for cte, increasing the D component tend to reduce the car's alignment with the curved tracks.

* I component reduces bias of the controls. As there is no obvious bias in the simulator, the effect of the I component is not as clear as the effects of the other two. However, during a long turning segments, the curvature of the road create bias (compared to the expectation of the controls), increasing I component helps keep the vehicle in the middle of the road, otherwise it has the tendency to drive more on one side. However, after the curved segments, the I component creates an inertia that keeps the vehicle on the opne side of the road for extended period of time


2. The final hyperparameters (P, I, D coefficients)

* I use twiddle to tune the parameters. I find it easier to tune P and D component first and I later, however in the submitted code I twiddle all three at the same time. To get the performance for a set of parameters, I run the simulations for 1000 "messages". I reset the simulator every time starting a new round fo twiddle.



