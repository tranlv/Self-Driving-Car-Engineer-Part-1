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

	for (int i = 0; i < num_particles; i++) { //c11 loop for vector
		Particle p;
		p.id = i;
		p.x = dist_x(gen); // add noise
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;

		particles.push_back(p);
		weights.push_back(p.weight);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.

	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/



	for (int i = 0; i< num_particles; i++) {
		double pred_x;
		double pred_y;
		double pred_theta;

		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;


		if (abs(yaw_rate) == 0) {
			pred_x = p_x + velocity * delta_t * cos(p_theta);
			pred_y = p_y + velocity * delta_t * sin(p_theta);
			pred_theta = p_theta;
		}
		else {
			pred_x = p_x + velocity * (sin(p_theta + yaw_rate * delta_t) - sin(p_theta))/yaw_rate ;
			pred_y = p_y + velocity * (cos(p_theta) - cos(p_theta + yaw_rate * delta_t))/yaw_rate ; 
			pred_theta = p_theta + delta_t * yaw_rate;
		}


		normal_distribution<double> dist_x(pred_x, std_pos[0]);
		normal_distribution<double> dist_y(pred_y, std_pos[1]);
		normal_distribution<double> dist_theta(pred_theta, std_pos[2]);

		// add noise
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta =  dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (int i = 0; i < observations.size(); i++) {
		
		double lowest_distance =  numeric_limits<double>::max();
		
		int map_id = -1;
		double obs_x = observations[i].x;
		double obs_y = observations[i].y;

		for (int j = 0; j < predicted.size(); j++) {
			double pred_x = predicted[j].x;
		 	double pred_y = predicted[j].y;
		  	int pred_id = predicted[j].id;

			double temp = dist(pred_x, pred_y,obs_x, obs_y);

			if (temp < lowest_distance) {
				lowest_distance =  temp;
				map_id = pred_id;
			}
		}

		observations[i].id = map_id;
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

	double weight_normalizer = 0.0;

	for (int i = 0; i < num_particles; i++) {		

		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;

		// coverting from vehicle to map system
		vector<LandmarkObs> transformed_os;
		for (int j = 0; j < observations.size(); j++) {
			double obs_x =  observations[j].x;
			double obs_y =  observations[j].y;
			int obs_id = observations[j].id;

			double x_map = p_x + (cos(p_theta) * obs_x) - (sin(p_theta) * obs_y);
			double y_map = p_y + (sin(p_theta) * obs_x) + (cos(p_theta) * obs_y);

			LandmarkObs landmark_obs;
			landmark_obs.x = x_map;
			landmark_obs.y = y_map;
			landmark_obs.id = obs_id;
			transformed_os.push_back(landmark_obs);
		}


		//  Filter map landmarks to keep those in sensor_range	

		vector<LandmarkObs> landmark_in_range;
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {

			float lm_x = map_landmarks.landmark_list[j].x_f;
			float lm_y = map_landmarks.landmark_list[j].y_f;
			int lm_id = map_landmarks.landmark_list[j].id_i;

			if(fabs(p_x - lm_x) <= sensor_range && fabs(p_y - lm_y) <= sensor_range) {
				// add to landmark in range
				LandmarkObs landmarkObs;
				landmarkObs.id = lm_id;
				landmarkObs.x = double(lm_x);
				landmarkObs.y = double(lm_y);
				landmark_in_range.push_back(landmarkObs);
			}
		}


		// Associate observations to lndmarks using nearest neighbor algorithm.
		dataAssociation(landmark_in_range, transformed_os);


		//reinit weight
		particles[i].weight = 1.0;

		double std_x = std_landmark[0];
		double std_y = std_landmark[1];
		double std_x_2 = std_x * std_x;
		double std_y_2 = std_y * std_y;
		double gaussian_norm = 1.0/(2.0 * M_PI * std_x * std_y);


		// Calculate the weight of each particle using Multivariate Gaussian distribution.*/
		for (int j = 0; j < transformed_os.size(); j++) {
			
			double obs_x = transformed_os[j].x;
			double obs_y = transformed_os[j].y;
			int obs_id = transformed_os[j].id;

			double exponent = 1.0;

			double pr_x, pr_y, pr_id;
			for (int k =0; k < landmark_in_range.size(); k++) {
				pr_x = landmark_in_range[k].x;
				pr_y = landmark_in_range[k].y;
				pr_id = landmark_in_range[k].id;

				if (obs_id == pr_id) {
					exponent = pow((pr_x - obs_x), 2)/(2.0 * std_x_2) + pow((pr_y - obs_y), 2)/(2.0 * std_y_2);
					particles[i].weight *= gaussian_norm * exp(-1.0 * exponent) ;
				}
			}
		}
		weight_normalizer += particles[i].weight;
	}

	for (int i = 0; i < particles.size(); i++) {
		particles[i].weight /= weight_normalizer;
		weights[i] = particles[i].weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	discrete_distribution<int> dist_particles(weights.begin(), weights.end());
	vector<Particle> resampled_particles;

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
