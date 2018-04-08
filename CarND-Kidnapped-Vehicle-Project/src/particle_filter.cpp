/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <cfloat>

#include "particle_filter.h"

using namespace std;


default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and othe[rs in this file).

	normal_distribution<double> dist_x(x, std[0])
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	num_particles = 100

	particles = vector<Particle>(num_particles);

	int id = 0;

	for (auto &p: particles) { //c11 loop for vector
		p.id = id;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1/100;
		id += 1; 
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.

	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/


	normal_distribution<double> dist_x(x, std_pos[0])
	normal_distribution<double> dist_y(y, std_pos[1]);
	normal_distribution<double> dist_theta(theta, std_pos[2]);

	for (auto &p: particles) {
		p.x = p.x + (velocity/yaw_rate)(sin(p.theta + yaw_rate * delta_t) - sin(p.theta)) + dist_x(gen);
		p.y = p.y + (velocity/yaw_rate)(cos(p.theta) - cos(theta + yaw_rate * delta_t)) + dist_y(gen);
		p.theta += p.theta * yaw_rate + dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (auto &obs : observations) {
		double distance = DBL_MAX;
		for (auto &pred: predicted) {
			double temp = dist(obs.x, obs.y, pred.x, pred.y);
			if (diff > temp) {
				diff =  temp;
				obs.id = landmark.id;
			}
		}
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	//	x_map= x_part + (np.cos(theta) * x_obs) - (np.sin(theta) * y_obs)

	for (auto &p: particles) {
		vector<LandmarkObs> landmark_in_range;
		int i= 0;
		for (auto &landmark: map_landmarks.landmark_list) {
			if(dist(landmark.x_f, landmark.y_f, p.x,p.y) <= sensor_range) {
				LandmarkObs landmarkObs;
				landmarkObs.id = i;
				landmarkObs.x = double(landmark.x_f);
				landmarkObs.y = double(landmark.y_f);
				landmark_in_range.push_back(landmarkObs);
			}
		}

		// coverting from vehicle to map system
		std::vector<LandmarkObs> transformed_obs = transform(observations, p);
		vector<LandmarkObs> transformed_obs;
		for (auto &obs: observations) {
			x_map = obs.x + (cos(p.theta) * obs.x) - (sin(p.theta) * obs.y);
			y_map= obs.y + (sin(theta) * obs.x) + (cos(theta) * obs.y)
			transformed_obs.x = x_map;
			transformed_obs.y = y_map;
		}

		dataAssociation(landmark_in_range, transformed_obs);

		double sum_sqr_x_diff = 0.0;
		double sum_sqr_y_diff = 0.0;

		for (auto &obs: transformed_obs) {
			double x_diff = obs.x - landmark_in_range[obs.id].x;
			double y_diff = obs.y - landmark_in_range[obs.id].y;
			sum_sqr_x_diff += x_diff * x_diff;
			sum_sqr_y_diff += y_diff * y_diff;
		}

		double std_x = std_landmark[0];
		double std_y = std_landmark[1];


		gaussian_norm = 1/(2 * M_PI *  std_x * std_y);
		exponent = sum_sqr_x_diff/(2 * std_x * std_x) + sum_sqr_y_diff/(2 * std_y * std_y);
		p.weight = gaussian_norm * (exp(-exponent)) ;

	}

	if  (weights.size() != num_particles) {
		weights = std::vector<double>(num_particles);
	}

	for (int i =0; i < num_particles; i++) {
		weights[i] = particles[i].weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::discrete_distribution<int> dist_particles(weights.begin(), weights.end());
	std::vector<Particle> resampled_particles;

	for (int i = 0; i <num_particles; i++) {
		resampled_particles.push_back(particles[dist_particles(gen)]);
	}

	particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
