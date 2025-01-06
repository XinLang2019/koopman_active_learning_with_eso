#ifndef TASK_TRAJ_HPP
#define TASK_TRAJ_HPP

#include <iostream>
#include <armadillo>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Geometry>

class Trajectory {
public:
    arma::vec task_trajectory(double theta) {
        // Parameters for the toroidal helix (torus knot)
        double R = 0.4;    // Major radius of the torus (distance from the center of the tube to the center of the torus)
        double r = 0.1;    // Minor radius of the torus (radius of the tube)
        double p = 0.05;   // Number of twists in the torus knot
        double q = 1;    // Number of loops through the hole of the torus
        arma::vec cycle_pose = {0.0, 0.0, 0.5};

        arma::vec pose = arma::zeros(7);

        // Parametric equations for the torus knot
        pose[0] = (R + r * std::cos(q * theta)) * std::cos(p * theta) + cycle_pose[0];
        pose[1] = -(R + r * std::cos(q * theta)) * std::sin(p * theta) + cycle_pose[1];
        pose[2] = r * std::sin(q * theta) + cycle_pose[2];
        
        // Euler angles
        pose[3] = std::asin(0.1 / std::sqrt(pose[0] * pose[0] + pose[1] * pose[1]));
        pose[4] = -1.57;
        pose[5] = p * theta;

        // Convert Euler angles to rotation vector using Armadillo functions
        // arma::mat rotation_matrix = arma::eye(3, 3);
        // rotation_matrix = rotx(pose[3]) * roty(pose[4]) * rotz(pose[5]);
        // arma::vec rotation_vector = arma::vectorise(rotation_matrix);

        Eigen::Vector3d euler_angles(pose[3], pose[4], pose[5]);
        Eigen::Quaterniond quaternion = Eigen::AngleAxisd(euler_angles[0], Eigen::Vector3d::UnitX())
                                        * Eigen::AngleAxisd(euler_angles[1], Eigen::Vector3d::UnitY())
                                        * Eigen::AngleAxisd(euler_angles[2], Eigen::Vector3d::UnitZ());

        pose[3] = quaternion.x();
        pose[4] = quaternion.y();
        pose[5] = quaternion.z();
        pose[6] = quaternion.w();
        // Replace pose[3:6] with the rotation vector
        // pose[3] = rotation_vector[0];
        // pose[4] = rotation_vector[1];
        // pose[5] = rotation_vector[2];

        arma::vec vel = arma::zeros(6);

        vel[0] = p * (R + r * std::cos(q * theta)) * std::sin(p * theta) + q * r * std::sin(q * theta) * std::cos(p * theta);
        vel[1] = -p * (R + r * std::cos(q * theta)) * std::cos(p * theta) + q * r * std::sin(q * theta) * std::sin(p * theta);
        vel[2] = q * r * std::cos(q * theta);
        vel[3] = 0.0;
        vel[4] = 0.0;
        vel[5] = 0.0;

        // Combine pose and velocity into a single vector for return
        arma::vec result(13);
        result.subvec(0, 6) = pose;
        result.subvec(7, 12) = vel;
        return result;
    }

    // 定义一个创建旋转矩阵的函数
    arma::mat rotx(double angle) {
        arma::mat R = arma::eye<arma::mat>(3, 3);
        R(1, 1) = std::cos(angle);
        R(1, 2) = -std::sin(angle);
        R(2, 1) = std::sin(angle);
        R(2, 2) = std::cos(angle);
        return R;
    }

    arma::mat roty(double angle) {
        arma::mat R = arma::eye<arma::mat>(3, 3);
        R(0, 0) = std::cos(angle);
        R(0, 2) = std::sin(angle);
        R(2, 0) = -std::sin(angle);
        R(2, 2) = std::cos(angle);
        return R;
    }

    arma::mat rotz(double angle) {
        arma::mat R = arma::eye<arma::mat>(3, 3);
        R(0, 0) = std::cos(angle);
        R(0, 1) = -std::sin(angle);
        R(1, 0) = std::sin(angle);
        R(1, 1) = std::cos(angle);
        return R;
    }
};

#endif