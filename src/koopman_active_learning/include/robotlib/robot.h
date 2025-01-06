#ifndef ROBOT_H
#define ROBOT_H

#include <ros/ros.h>
#include <intera_core_msgs/SolvePositionIK.h>
#include <intera_core_msgs/SolvePositionIKRequest.h>
#include <gazebo_msgs/LinkStates.h>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Quaternion.h>

#include <std_msgs/Header.h>
#include <sensor_msgs/JointState.h>
#include <intera_core_msgs/JointCommand.h>
#include <intera_core_msgs/EndpointState.h>
#include <franka_msgs/ModelState.h>
#include <franka_gripper/GraspActionGoal.h>
#include <franka_gripper/MoveActionGoal.h>

#include <math.h>
#include <armadillo>

#include "dynamicalSystems/system.hpp"
#include "dynamicalSystems/doubleintegrator.hpp"
#include "dynamicalSystems/koopman/KoopmanSystem.hpp"
#include "dynamicalSystems/koopman/basis_functions/basis.hpp"
#include "dSAClib/SAC.hpp"
#include "dSAClib/objective.hpp"

#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <string>
#include <sstream>

#include <kdl/chainiksolverpos_lma.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chain.hpp>
#include <kdl/jntarray.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/frames_io.hpp> 

// #include "dSAClib/SAC_trajectory.hpp"
// #include "dSAClib/trajectory_objective.hpp"

// #include "dSAClib/wrap2Pi.hpp"
typedef RigidBodyBasis BasisFun;

class SawyerRobot {

private:

    std_msgs::Header _hdr;
    geometry_msgs::Point _p1;
    geometry_msgs::Quaternion _q1;
    geometry_msgs::Pose _pose;
    geometry_msgs::PoseStamped _pose_stamped;


    intera_core_msgs::SolvePositionIKRequest _ik_req;
    intera_core_msgs::SolvePositionIK _ik_srv;
    intera_core_msgs::JointCommand _command_joints;
    franka_gripper::GraspActionGoal _command_gripper;
    franka_gripper::MoveActionGoal _command_gripper_stop;

    franka_msgs::ModelState _model_states;

    KDL::Tree my_tree;
    std::string robot_desc_string;
    KDL::Chain chain;
    KDL::JntArray joint_positions;

    std::vector<std::string> joint_names = {"right_j0", "right_j1", "right_j2", "right_j3", "right_j4", "right_j5", "right_j6"};
    arma::vec initial_joint_positions = {0, 0.7,-M_PI, 2.5, M_PI/2, -M_PI/2 , 0};

    arma::vec joint_limits = {165, 100, 165, 176, 165, 170, 165};
    // arma::vec joint_limits = {350, 350, 350, 350, 341, 341, 540};
    float joint_limit_value = 4;
    arma::vec P_joint_vals = {30.0, 80.0, 15.0, 60.0, 13.0, 15.0, 1.5};
    arma::vec D_joint_vals = {2, 8, 0.1, 1, 1, 1, 0.1};
    arma::vec I_joint_vals = {0.4, 0.8, 0.4, 0.4, 0.1, 2, 0.1};
    arma::vec _err = arma::zeros<arma::vec>(7);
    arma::vec _prev_err = arma::zeros<arma::vec>(7);
    arma::vec _integrated_err = arma::zeros<arma::vec>(7);

    float _time_step = 0;


public:

    std::string ik_ns = "ExternalTools/right/PositionKinematicsNode/IKService";
    std::string robot_ns = "robot/limb/right/";
    ros::Publisher _joint_cmd_pub;
    ros::Publisher _gripper_cmd_pub;
    ros::Publisher _gripper_stop_cmd_pub;
    ros::Subscriber _joint_state_sub;
    ros::Subscriber _model_state_sub;
    ros::Subscriber _endpoint_state_sub;
    ros::Subscriber _ball_state_sub;
    ros::ServiceClient _ik_client;

    ros::Rate* loop_rate;

    bool hasInitialized = false;
    arma::vec endEffector_pose;
    arma::vec endEffector_vel;
    arma::vec endEffector_twist;
    arma::vec endEffector_wrench;
    arma::vec joint_poses;
    arma::vec joint_velocities;
    arma::vec joint_torque;
    arma::vec joint_torque_grav;
    
    arma::vec control_input;
    arma::vec jacobian;

    /* Controller stuff */

    // system dynamics 
    System* sys;
    arma::vec _dynamics_state;
    arma::vec _ball_pose;
    arma::vec _ball_velocities_filter;
    arma::vec _ball_velocities;
    KoopmanSystem* ksys;
    deiSAC* controller;


    SawyerRobot(ros::Rate* _loop_rate, float time_step) { // Constructor

        loop_rate = _loop_rate;
        ros::NodeHandle nh;
        _time_step = time_step;
        joint_limits *= M_PI/180.0;
        endEffector_pose = arma::zeros<arma::vec>(7);
        endEffector_vel = arma::zeros<arma::vec>(7);
        endEffector_twist = arma::zeros<arma::vec>(6);
        endEffector_wrench = arma::zeros<arma::vec>(6);

        _dynamics_state = {0.,0.,0.,0.};
        _ball_pose = {0.,0.,0.};

        _ball_velocities = {0.,0.,0.};
        _ball_velocities_filter = {0.,0.,0.};
        joint_poses = arma::zeros<arma::vec>(7);
        joint_velocities = arma::zeros<arma::vec>(7);
        control_input = arma::zeros<arma::vec>(7);
        joint_torque  = arma::zeros<arma::vec>(7);
        joint_torque_grav  = arma::zeros<arma::vec>(7);
        jacobian = arma::zeros<arma::vec>(42);

        this->_ik_client = nh.serviceClient<intera_core_msgs::SolvePositionIK>(this->ik_ns);
        this->_joint_cmd_pub = nh.advertise<intera_core_msgs::JointCommand>( "/joint_effort_example_controller/joint_command", 1);
        this->_gripper_cmd_pub = nh.advertise<franka_gripper::GraspActionGoal>( "/franka_gripper/grasp/goal", 1);
        this->_gripper_stop_cmd_pub = nh.advertise<franka_gripper::MoveActionGoal>( "/franka_gripper/move/goal", 1);

        this->_joint_state_sub = nh.subscribe("/joint_states", 1, &SawyerRobot::getJointState, this);
        this->_model_state_sub = nh.subscribe("/joint_effort_example_controller/model_states", 1, &SawyerRobot::getModelState, this);
        this->_endpoint_state_sub = nh.subscribe("/gazebo/link_states", 1, &SawyerRobot::getEndState, this);
        // this->_ball_state_sub = nh.subscribe("ball_tracker", 1, &SawyerRobot::getBallState, this);
        
        {
            arma::vec x = {0,0,0,0,0,0,0};
            this->setRobotToInitialJointPosition(x);
            hasInitialized = true;
        }

        /* Set up the operator */
        sys = new DoubleIntegrator(time_step);
        ksys = new KoopmanSystem(new BasisFun());

        arma::vec Qdiag = 200 * arma::ones<arma::vec>(14);

        Qdiag.tail_rows(7) = 2 * arma::ones<arma::vec>(7);

        arma::vec Rdiag = 0.1 * arma::ones<arma::vec>(7);

        arma::mat _Q = arma::diagmat(Qdiag);
        arma::mat _R = arma::diagmat(Rdiag);

        arma::vec _Qfdiag = 0.01*Qdiag;
        _Qfdiag.tail_rows(7) = 1.0*arma::ones<arma::vec>(7);
        arma::mat _Qf = arma::diagmat(_Qfdiag);

        arma::vec umax = 2 * arma::ones<arma::vec>(7);
        umax[0] = 5;
        umax[1] = 5;
        umax[2] = 5;
        umax[3] = 5;
        arma::vec unom = 0.01 * arma::ones<arma::vec>(7);
        umax[1] = 0.1;
        arma::vec desired_state = arma::zeros<arma::vec>(14);

        const float T = 1.5;
        const int N = 50;// (int)(T/time_step);

        controller = new deiSAC(ksys, new Objective(_Q, _R, _Qf, desired_state, new BasisFun()), N, umax, unom);

        nh.param("/robot_description", robot_desc_string, std::string());
        if (!kdl_parser::treeFromString(robot_desc_string, my_tree)) {
            ROS_ERROR("Failed to construct kdl tree");
        }
        if (!my_tree.getChain("panda_link0", "panda_link7", chain)) {
            ROS_ERROR("Failed to get chain from panda_link0 to panda_link7");
        }
    }

    ~SawyerRobot() {
        forwardTorqueCommandState(arma::zeros<arma::vec>(7));//delete loop_rate ? ;
    };

    void switchWeights(bool velWeight = true) {
        controller->obj->q_fisher = 0;
        arma::vec Qdiag = 50 * arma::ones<arma::vec>(14);
        arma::vec _Qfdiag = 1*Qdiag;
        // Qdiag[4] = 300;
        // Qdiag[5] = 300;
        // Qdiag[6] = 300;
        if (velWeight) {
            Qdiag.tail_rows(7) = 1 * arma::ones<arma::vec>(7);
        } else {
            // more aggresive trajectories
            // Qdiag.tail_rows(7) = 0.001 * arma::ones<arma::vec>(7);
            // less agressive trajectories
            Qdiag.tail_rows(7) = 0.0 * arma::ones<arma::vec>(7);
            _Qfdiag.tail_rows(7) = 0.0 * arma::ones<arma::vec>(7);
            
        }
        arma::mat Q = arma::diagmat(Qdiag);
        arma::mat _Qf = arma::diagmat(_Qfdiag);
        // more aggresive trajectories
        // arma::vec Rdiag = 0.001 * arma::ones<arma::vec>(7);
        arma::vec Rdiag = 0.01 * arma::ones<arma::vec>(7);
        arma::mat R = arma::diagmat(Rdiag);
        controller->obj->Q = Q;
        controller->obj->R = R;
        controller->obj->Qf = _Qf;
    }

    arma::vec computeJointTorques(const arma::vec& desired_joint_pose) {
        _err = desired_joint_pose - joint_poses;
        _integrated_err += _err * _time_step;
        arma::vec computedTorques = P_joint_vals % _err - D_joint_vals % joint_velocities + I_joint_vals % _integrated_err;
        return computedTorques;
    }

    arma::vec checkJointLimits() {
        double eps = 0.2;
        arma::vec joint_limit_torque = arma::zeros<arma::vec>(7);
        for (int i = 0; i < joint_poses.n_rows; i++) {
            if (i == 5){
                if (joint_poses[i] > 3.75-eps) {
                joint_limit_torque[i] = - joint_limit_value;
                } else if (joint_poses[i] < 0.0+eps) {
                joint_limit_torque[i] = joint_limit_value;
                }
            }
            else{
                if (joint_poses[i] > joint_limits[i]-eps) {
                joint_limit_torque[i] = - joint_limit_value;
                } else if (joint_poses[i] < - joint_limits[i]+eps) {
                    joint_limit_torque[i] = joint_limit_value;
                }
            }
            
        }
        return joint_limit_torque;
    }

    void getEndpointState(const intera_core_msgs::EndpointState::ConstPtr& msg) {
        endEffector_pose[0] = msg->pose.position.x;
        endEffector_pose[1] = msg->pose.position.y;
        endEffector_pose[2] = msg->pose.position.z;
        endEffector_pose[3] = msg->pose.orientation.x;
        endEffector_pose[4] = msg->pose.orientation.y;
        endEffector_pose[5] = msg->pose.orientation.z;
        endEffector_pose[6] = msg->pose.orientation.w;

        endEffector_twist[0] = msg->twist.linear.x;
        endEffector_twist[1] = msg->twist.linear.y;
        endEffector_twist[2] = msg->twist.linear.z;
        endEffector_twist[3] = msg->twist.angular.x;
        endEffector_twist[4] = msg->twist.angular.y;
        endEffector_twist[5] = msg->twist.angular.z;

        endEffector_wrench[0] = msg->wrench.force.x;
        endEffector_wrench[1] = msg->wrench.force.y;
        endEffector_wrench[2] = msg->wrench.force.z;
        endEffector_wrench[3] = msg->wrench.torque.x;
        endEffector_wrench[4] = msg->wrench.torque.y;
        endEffector_wrench[5] = msg->wrench.torque.z;

        //std::cout << endEffector_state << std::endl;
        _dynamics_state[0] = msg->pose.position.y;
        _dynamics_state[1] = msg->pose.position.z;
        _dynamics_state[2] = msg->twist.linear.y;
        _dynamics_state[3] = msg->twist.linear.z;

    }

    void getEndState(const gazebo_msgs::LinkStates::ConstPtr& msg) {
        endEffector_pose[0] = msg->pose[8].position.x;
        endEffector_pose[1] = msg->pose[8].position.y;
        endEffector_pose[2] = msg->pose[8].position.z;
        endEffector_pose[3] = msg->pose[8].orientation.x;
        endEffector_pose[4] = msg->pose[8].orientation.y;
        endEffector_pose[5] = msg->pose[8].orientation.z;
        endEffector_pose[6] = msg->pose[8].orientation.w;

    }

    void getBallState(const geometry_msgs::Point::ConstPtr& msg) {


        _ball_velocities_filter[0] = msg->x - _ball_pose[0];
        _ball_velocities_filter[1] = msg->y - _ball_pose[1];

        _ball_velocities = 0.8 * _ball_velocities + (1 - 0.8) * _ball_velocities_filter;

        _ball_velocities[2] = (msg->y * _ball_velocities[0] - msg->x * _ball_velocities[1])/ (msg->x * msg->x + msg->y * msg->y);
        
        _ball_pose[0] = msg->x;
        _ball_pose[1] = msg->y;
        _ball_pose[2] = atan2( _ball_pose[0] ,  _ball_pose[1] );
        // std::cout << _ball_velocities << std::endl;
    }

    void getJointState(const sensor_msgs::JointState::ConstPtr& msg ) {
        // arma::vec velocity_filter = arma::zeros<arma::vec>(7);
        // arma::vec position_filter = arma::zeros<arma::vec>(7);
        for (int i = 0; i < 7; i++) {
            // joint_poses[i] = msg->position[i];
            // joint_velocities[i] = msg->velocity[i];
            // joint_torque[i] = msg->effort[i];
            // position_filter[i-1] = msg->position[i];
            // velocity_filter[i-1] = msg->velocity[i];
        }

        // joint_velocities = 0.2 * joint_velocities + (1 - 0.2) * velocity_filter;
        // joint_poses = 0.2 * joint_poses + (1-0.2) * position_filter;
        // std::cout << joint_poses << std::endl;
        // std::cout << joint_velocities << std::endl;
    }

    void getModelState(const franka_msgs::ModelState::ConstPtr& msg ) {
        for (int i = 0; i < 7; i++) {
            joint_poses[i] = msg->q[i];
            joint_velocities[i] = msg->dq[i];
            joint_torque[i] = msg->torque[i];
            joint_torque_grav[i] = msg->gravity[i];
        }
        
        for (int i = 0; i < 42; i++) {
            jacobian[i] = msg->jacobian[i];
        }
    }

    arma::vec getDataState() {
        return arma::join_cols(joint_poses, joint_velocities);
    }
    arma::mat getJacobian(){
        return arma::reshape(jacobian, 6, 7);
    }
   
    void setRobotToInitialJointPosition(const arma::vec& x) {
        float _tic = 0.0;
        float _toc = 10.0;
        float _dt = 1.0/100.0;
        while ( ros::ok() ) {
            forwardPositionCommandState(x);
            ros::spinOnce();
            loop_rate->sleep();
            _tic += _dt;
            if (_tic > _toc) { break; }
        }
    }

    void setRobotToInitialPosition(const arma::vec& x) {
        _hdr.stamp = ros::Time::now();
        _hdr.frame_id = "base";

        _p1.x = x[0];
        _p1.y = x[1];
        _p1.z = x[2];

        _q1.x = x[3];
        _q1.y = x[4];
        _q1.z = x[5];
        _q1.w = x[6];

        _pose.position = _p1;
        _pose.orientation = _q1;

        _pose_stamped.header = _hdr;
        _pose_stamped.pose = _pose;


        _ik_req.pose_stamp.push_back(_pose_stamped);
        _ik_req.tip_names.push_back("right_hand");
        _ik_req.seed_mode = intera_core_msgs::SolvePositionIKRequest::SEED_CURRENT;

        _ik_srv.request = _ik_req;

        std::cout << "Setting Sawyer to starting position ..." << std::endl;
        if (_ik_client.call(this->_ik_srv)) {
            _command_joints.mode = intera_core_msgs::JointCommand::POSITION_MODE;
            _command_joints.header = _hdr;
            _command_joints.names = _ik_srv.response.joints[0].name;
            joint_names = _ik_srv.response.joints[0].name;
            _command_joints.position = _ik_srv.response.joints[0].position;

            float _tic = 0.0;
            float _toc = 4.0;
            float _dt = 1.0/100.0;
            while ( ros::ok() ) {
                _joint_cmd_pub.publish(_command_joints);
                ros::spinOnce();
                loop_rate->sleep();
                _tic += _dt;
                if (_tic > _toc) { break; }
                else if ( arma::norm(x.rows(0,2) - endEffector_pose.rows(0,2)) < 0.01 ) {
                    break;
                }
            }
            std::cout << "Should be at the position now!" << std::endl;
        } else {
            std::cout << "Could not find starting position ..." << std::endl;
        }
    }

    bool forwardTorqueCommandState(const arma::vec& u) {
        _hdr.stamp = ros::Time::now();
        _command_joints.header = _hdr;
        _command_joints.names = joint_names;
        _command_joints.mode = intera_core_msgs::JointCommand::TORQUE_MODE;
        _command_joints.effort.resize(7);
        for (int i = 0; i < _command_joints.effort.size(); i++) {
            _command_joints.effort[i] = u[i];
        }
        // std::cout << u << std::endl;
        _joint_cmd_pub.publish(_command_joints);

        return true;
    }

    bool GripperCommandGet() {
          // capture
        _command_gripper.goal.width = 0.03; //capture body width
        _command_gripper.goal.epsilon.inner = 0.005;
        _command_gripper.goal.epsilon.outer = 0.005;
        _command_gripper.goal.speed = 0.1;
        _command_gripper.goal.force = 5; 
        
        _gripper_cmd_pub.publish(_command_gripper);

        return true;
    }

    bool GripperCommandStop() {
          // capture
        _command_gripper_stop.goal.speed = 0.1;
        _command_gripper_stop.goal.width = 0.05;
        _gripper_stop_cmd_pub.publish(_command_gripper_stop);

        return true;
    }

    bool endCommandState(const arma::vec& u) {
        _hdr.stamp = ros::Time::now();
        _command_joints.header = _hdr;
        _command_joints.names = joint_names;
        _command_joints.mode = intera_core_msgs::JointCommand::POSITION_MODE;
        _command_joints.effort.resize(7);
        for (int i = 0; i < _command_joints.effort.size(); i++) {
            _command_joints.effort[i] = u[i];
        }
        // std::cout << u << std::endl;
        _joint_cmd_pub.publish(_command_joints);

        return true;
    }

    bool forwardTorqueCommandStateIndividual( double u , int joint_idx) {
        _hdr.stamp = ros::Time::now();
        _command_joints.header = _hdr;
        _command_joints.names.resize(1);
        _command_joints.names[0] = joint_names[joint_idx];
        _command_joints.names = joint_names;
        _command_joints.mode = intera_core_msgs::JointCommand::TORQUE_MODE;
        _command_joints.effort.resize(1);
        _command_joints.effort[0] = u;
        // std::cout << u << std::endl;
        _joint_cmd_pub.publish(_command_joints);

        return true;
    }

    bool forwardVelocityCommandState(const arma::vec& u) {
        _hdr.stamp = ros::Time::now();
        _command_joints.header = _hdr;
        _command_joints.names = joint_names;
        _command_joints.mode = intera_core_msgs::JointCommand::VELOCITY_MODE;
        _command_joints.velocity.resize(7);
        for (int i = 0; i < _command_joints.velocity.size(); i++) {
            _command_joints.velocity[i] = u[i];
        }
        // std::cout << u << std::endl;
        _joint_cmd_pub.publish(_command_joints);

        return true;
    }


    bool forwardPositionCommandState(const arma::vec& u) {
        _hdr.stamp = ros::Time::now();
        _command_joints.header = _hdr;
        _command_joints.names = joint_names;
        _command_joints.mode = intera_core_msgs::JointCommand::POSITION_MODE;
        _command_joints.position.resize(7);
        for (int i = 0; i < _command_joints.position.size(); i++) {
            _command_joints.position[i] = u[i];
        }
        // std::cout << u << std::endl;
        _joint_cmd_pub.publish(_command_joints);

        return true;
    }

    arma::vec solveIK(arma::vec& x) {
        KDL::ChainIkSolverPos_LMA ik_solver(chain); // Inverse kin. solver
        // KDL::Frame desired_pose(KDL::Rotation::Identity(), KDL::Vector(x[0], x[1], x[2]));  
        KDL::Frame desired_pose(KDL::Rotation::Quaternion(x[3], x[4], x[5], x[6]), KDL::Vector(x[0], x[1], x[2]));

        // Initialize joint_positions with the number of joints in the chain
        joint_positions.resize(chain.getNrOfJoints());
        arma::vec target_states = arma::zeros<arma::vec>(joint_positions.data.size());
        // Perform the IK solver
        int ret = ik_solver.CartToJnt(joint_positions, desired_pose, joint_positions);
        if (ret < 0) {
            ROS_ERROR("IK solver failed");
            return target_states;
        }
        else{
            for (unsigned int i = 0; i < joint_positions.data.size(); i++) {
                target_states[i] = joint_positions(i);
                // ROS_INFO("Joint %d: %f", i, joint_positions(i));
            }
        }
        return target_states;
    }

    arma::vec getIKState(const arma::vec& x) {
        _hdr.stamp = ros::Time::now();
        _hdr.frame_id = "base";

        _p1.x = x[0];
        _p1.y = x[1];
        _p1.z = x[2];

        _q1.x = x[3];
        _q1.y = x[4];
        _q1.z = x[5];
        _q1.w = x[6];

        _pose.position = _p1;
        _pose.orientation = _q1;

        _pose_stamped.header = _hdr;
        _pose_stamped.pose = _pose;

        _ik_req.pose_stamp.resize(1);
        _ik_req.tip_names.resize(1);
        _ik_req.pose_stamp[0] = _pose_stamped;
        _ik_req.tip_names[0] = "right_hand";
        _ik_req.seed_mode = intera_core_msgs::SolvePositionIKRequest::SEED_NS_MAP;

        _ik_srv.request = _ik_req;

        arma::vec target_states = arma::zeros<arma::vec>(7);
        if (_ik_client.call(this->_ik_srv)) {
            _command_joints.header = _hdr;
            _command_joints.names = _ik_srv.response.joints[0].name;

            _command_joints.position = _ik_srv.response.joints[0].position;
            for (int i = 0; i < _command_joints.position.size(); i++) {
                target_states[i] = _command_joints.position[i];
            }
        }
        return target_states;
    }

    bool getAndSetIKState(const arma::vec& x) {
        _hdr.stamp = ros::Time::now();

        _pose.position.x = x[0];
        _pose.position.y = x[1];
        _pose.position.z = x[2];

        _pose.orientation.x = x[3];
        _pose.orientation.y = x[4];
        _pose.orientation.z = x[5];
        _pose.orientation.w = x[6];

        _pose_stamped.header = _hdr;
        _pose_stamped.pose = _pose;


        _ik_req.pose_stamp[0] = _pose_stamped;

        _ik_srv.request = _ik_req;

        if (_ik_client.call(this->_ik_srv)) {
            _command_joints.header = _hdr;
            _command_joints.names = _ik_srv.response.joints[0].name;

            _command_joints.position = _ik_srv.response.joints[0].position;
            for (int i = 0; i < _command_joints.position.size(); i++) {
                control_input[i] = _command_joints.position[i];
            }
            _joint_cmd_pub.publish(_command_joints);
            return true;
        }
        return false;
    }

    Eigen::MatrixXd loadTxtData(const std::string& filename, int rows, int cols) {
        Eigen::MatrixXd data(rows, cols);

        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open the file." << std::endl;
            return Eigen::MatrixXd();
        }

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                if (!(file >> data(i, j))) {
                    std::cerr << "Error reading data from file." << std::endl;
                    return Eigen::MatrixXd();
                }
            }
        }

        file.close();

        return data;
    }

};


#endif
