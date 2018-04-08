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

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	num_particles = 100;

	particles = vector<Particle>(num_particles);

	int id = 0;

	for (auto &p: particles) { //c11 loop for vector
		p.id = id;
		p.x = x + dist_x(gen); // add noise
		p.y = y + dist_y(gen);
		p.theta = theta + dist_theta(gen);
		p.weight = 1.0;
		id += 1; 
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.

	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/


	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	for (auto &p: particles) {
		if (fabs(yaw_rate) < 0.00001) {
			p.x += velocity * delta_t * cos(p.theta);
			p.y += velocity * delta_t * sin(p.theta);
		}
		else {
			p.x +=  velocity * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta))/yaw_rate ;
			p.y +=  velocity * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t))/yaw_rate ; 
			p.theta += p.theta * yaw_rate;
		}

		// add noise
		p.x += dist_x(gen);
		p.y += dist_y(gen);
		p.theta +=  dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (auto &obs : observations) {
		double distance =  numeric_limits<double>::max();
		int map_id = -1;
		for (auto &pred: predicted) {

			double temp = dist(obs.x, obs.y, pred.x, pred.y);
			if (distance > temp) {
				distance =  temp;
				map_id = pred.id;
			}
		}

		obs.id = map_id;

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


	for (auto &p: particles) {		

		double p_x = p.x;
		double p_y = p.y;
		double p_theta = p.theta;


		vector<LandmarkObs> landmark_in_range;
		int i= 0;
		for (auto &landmark: map_landmarks.landmark_list) {

			float lm_x = landmark.x_f;
			float lm_y = landmark.y_f;
			int lm_id = landmark.id_i;

			if(dist(landmark.x_f, landmark.y_f, p.x,p.y) <= sensor_range) {
				// add to landmark in range
				LandmarkObs landmarkObs;
				landmarkObs.id = lm_id;
				landmarkObs.x = double(lm_x);
				landmarkObs.y = double(lm_y);
				landmark_in_range.push_back(landmarkObs);
			}
		}

		// coverting from vehicle to map system
		vector<LandmarkObs> transformed_os;
		for (auto &obs: observations) {
			double x_map = p.x + (cos(p.theta) * obs.x) - (sin(p.theta) * obs.y);
			double y_map= p.y + (sin(p.theta) * obs.x) + (cos(p.theta) * obs.y);

			LandmarkObs landmark_obs;
			landmark_obs.x = x_map;
			landmark_obs.y = y_map;
			landmark_obs.id = obs.id;
			transformed_os.push_back(landmark_obs);
		}

		dataAssociation(landmark_in_range, transformed_os);

		//reinit weight
		p.weight = 1.0;

		for (auto &obs: transformed_os) {
			double o_x = obs.x;
			double o_y = obs.y;
			int o_id = obs.id;

			double pr_x, pr_y;
			for (auto &lm_in_range: landmark_in_range) {
				if (lm_in_range.id == o_id) {
					pr_x = lm_in_range.x;
					pr_y = lm_in_range.y;
				}
			}

			double std_x = std_landmark[0];
			double std_y = std_landmark[1];
 
			double gaussian_norm = 1/(2 * M_PI *  std_x * std_y);
			double exponent = pow(pr_x-o_x,2)/(2 * std_x * std_x) + pow(pr_y-o_y,2)/(2 * std_y * std_y);
			p.weight *= gaussian_norm * (exp(-exponent)) ;
		}

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
