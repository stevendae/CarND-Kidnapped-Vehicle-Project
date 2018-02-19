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

#include "particle_filter.h"

#define zero 0.00001

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	this -> num_particles = 100;
	this -> particles.resize(num_particles);
	this -> weights.resize(num_particles, 1.0);
	default_random_engine gen;

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	unsigned int id_= 0;

	for (auto &p:particles){
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;
		p.id = id_;
		id_ += 1;
	}

	this -> is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;

	normal_distribution<double> norm_x(0, std_pos[0]);
	normal_distribution<double> norm_y(0, std_pos[1]);
	normal_distribution<double> norm_theta(0, std_pos[2]);


	for (auto &p:particles){
		if (fabs(yaw_rate) < zero) {
			p.x += velocity * delta_t * cos(p.theta);
			p.y += velocity * delta_t * sin(p.theta);
		}
		else {
			p.x += (velocity / yaw_rate) * (sin(p.theta + yaw_rate*delta_t) - sin(p.theta));
			p.y += (velocity / yaw_rate) * (cos(p.theta) - cos(p.theta + yaw_rate*delta_t));
			p.theta += yaw_rate*delta_t;
		}

		p.x += norm_x(gen);
		p.y += norm_y(gen);
		p.theta += norm_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (auto &o:observations) {

		double min_dist = std::numeric_limits<double>::max();
		for (auto p:predicted) {
			double test_dist = dist(o.x, o.y, p.x, p.y);
			if (test_dist < min_dist) {
				min_dist = test_dist;
				o.id = p.id;
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

	for (auto &p:particles){

		// identify which landmarks are valid for each particle; within sensor range
		std::vector <LandmarkObs> valid_land;
		for (auto &land:map_landmarks.landmark_list){
			double valid_dist = dist(p.x, p.y, land.x_f, land.y_f);
			if (valid_dist <= sensor_range){
				valid_land.push_back(LandmarkObs{land.id_i, land.x_f, land.y_f});
			}
		}

		std::vector<LandmarkObs> map_observations;
		for (const auto &o:observations){
			double x_m = o.x;
			double y_m = o.y;
			double theta_ = p.theta;

			LandmarkObs land_m;

			land_m.x = p.x + (cos(theta_)*x_m) - (sin(theta_)*y_m);
			land_m.y = p.y + (sin(theta_)*x_m) + (cos(theta_)*y_m);

			map_observations.push_back(land_m);
		}

		dataAssociation(valid_land, map_observations);

		p.weight = 1.0;

		for (auto &map_o:map_observations){
			double prob;

			for (auto &v_land:valid_land){
				if (map_o.id == v_land.id){
					double x_diff = map_o.x - v_land.x;
					double y_diff = map_o.y - v_land.y;

					double x_comp = pow(x_diff, 2.0) / (2.0 * pow(std_landmark[0], 2));
					double y_comp = pow(y_diff, 2.0) / (2.0 * pow(std_landmark[1], 2));

					double exp_ = exp(-1.0*(x_comp + y_comp));

					double norm_ = 1.0 / (2.0 * M_PI * std_landmark[0]*std_landmark[1]);

					prob = norm_ * exp_;
					break;
				}
			}
			if (prob < zero){
				prob = zero;
			}
			p.weight *= prob;
		}
		weights.push_back(p.weight);
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::vector<Particle> resampled;
	default_random_engine gen;
	std::discrete_distribution<int> sample_wheel(weights.begin(), weights.end());
	std::uniform_real_distribution<double> beta_increment(0, *std::max_element(weights.begin(), weights.end()));

	int index = sample_wheel(gen);
	double beta = 0;

	for (int i = 0; i < num_particles; i++) {
		beta += beta_increment(gen);
		while (weights[index] < beta){
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
		resampled.push_back(particles[index]);
	}

	this -> particles = resampled;
	weights.clear();

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
