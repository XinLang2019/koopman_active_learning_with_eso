#include <ros/ros.h>
#include "robotlib/robot.h"
#include <intera_core_msgs/SolvePositionIK.h>
#include <intera_core_msgs/SolvePositionIKRequest.h>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Quaternion.h>

#include <std_msgs/Header.h>
#include <sensor_msgs/JointState.h>

#include <intera_core_msgs/JointCommand.h>

#include <math.h>

#include "robotlib/dynamicalSystems/koopman/KoopmanSystem.hpp"
#include "robotlib/dynamicalSystems/koopman/basis_functions/basis.hpp"
#include <ctime>
#include <std_msgs/Empty.h>

#include "robotlib/lqr_controller.hpp"
#include "sawyer_data_torque_control/Koopman2Online.h"

int main(int argc, char** argv) {
    ros::init(argc, argv, "check_ik_solver");

    float filter_value = 0.5;
    ros::NodeHandle nh;
    ros::Rate loop_rate(100);
    float _tcurr = 0.0;
    float _dt = 1.0/100.0;

    SawyerRobot robot(&loop_rate, _dt);

    arma::vec state = robot.getDataState();
    arma::vec target_ee_state = { 0.6, 0.0, 0.3, 0.5, 0.5, 0.5, 0.5 };
    arma::vec target_joint_state = robot.getIKState(target_ee_state);
    arma::vec homeing_joint_state = {0,0,0, 0,0,0,0};

    std::vector<arma::vec> joint_state_collector;
    std::vector<arma::vec> ee_state_collector;
    std::vector<arma::vec> target_ee_collector;
    std::vector<arma::vec> target_joint_collector;
    std::vector<arma::vec> torque_controller_collector;
    std::vector<arma::vec> pose_error_collector;
    std::vector<arma::vec> end_pose_collector;
    std::vector<arma::vec> des_pose_collector;
    std::vector<arma::vec> koopman_collector;
    std::vector<arma::vec> u_collector;
    std::vector<arma::vec> xd_collector;

    std::string filePath = "/home/master/milab/franka_ws/src/sawyer_data_torque_control/data/";
    arma::mat joint_state_data;
    arma::mat ee_state_data;
    arma::mat target_ee_data;
    arma::mat target_joint_data;
    arma::mat torque_data;
    arma::mat loss_data;
    arma::mat end_pose_data;
    arma::mat des_pose_data;
    arma::mat koopman_data;
    arma::mat u_data;
    arma::mat xd_data;

    arma::mat jacobian;

    arma::vec command_torque = arma::zeros<arma::vec>(7);

    // ----------- System ID ----------- //
    std::cout << "Active Learning start ... \n";
    float timer = 0;
    robot.controller->obj->q_fisher = 50;
    target_ee_state = { 0.554, 0.0, 0.62, 1.0, 0.0, 0.0, 0.0 };
    target_joint_state = robot.solveIK(target_ee_state);
    robot.controller->obj->xd.head_rows(7) = target_joint_state;

    arma::vec safty_torques = arma::zeros<arma::vec>(7);
    arma::vec cdataIn = arma::zeros<arma::vec>(7);
    arma::vec tor = arma::zeros<arma::vec>(7);
    arma::vec dataIn = robot.getDataState();
    arma::vec dataOut;
    while (ros::ok() && timer < 20) {
        cdataIn = robot.controller->get_control(dataIn);
        command_torque = filter_value * command_torque + (1-filter_value)*cdataIn;
        safty_torques = robot.checkJointLimits();
        command_torque += safty_torques; 
        //command_torque=tor; //test
        robot.forwardTorqueCommandState(command_torque);
        // std::cout << "dataIn:" << command_torque.t() << std::endl;
    
        timer += _dt;        
        loop_rate.sleep();
        ros::spinOnce();  
        dataOut = robot.getDataState();
        robot.ksys->gradStep(dataIn, cdataIn, dataOut);
        dataIn = dataOut;

        joint_state_collector.push_back(robot.getDataState());
        u_collector.push_back(command_torque);
        end_pose_collector.push_back(robot.endEffector_pose);
        xd_collector.push_back(robot.controller->obj->xd.head_rows(7));
      
    }    

    // ----------------- Home to zero ----------------------------- //
    robot.controller->umax[1] = 10;
    // robot.controller->umax[5] = 10;
    // robot.controller->umax[4] = 10;
    robot.controller->obj->q_fisher = 0;
    robot.controller->obj->xd.head_rows(7) = target_joint_state;
    robot.switchWeights();
    std::cout << "Learning end and go to goal ... \n";
    while ( ros::ok() && _tcurr < 10) {
        cdataIn = robot.controller->get_control(dataIn);
        command_torque = filter_value * command_torque + (1-filter_value)*cdataIn;
        safty_torques = robot.checkJointLimits();
        command_torque += safty_torques; 
        
        robot.forwardTorqueCommandState(command_torque);
        loop_rate.sleep();
        _tcurr += _dt;
        ros::spinOnce();
        dataOut = robot.getDataState();
        robot.ksys->gradStep(dataIn, cdataIn, dataOut);
        dataIn = dataOut;

        // get and store data
        joint_state_collector.push_back(robot.getDataState());
        u_collector.push_back(command_torque);
        end_pose_collector.push_back(robot.endEffector_pose);
        xd_collector.push_back(robot.controller->obj->xd.head_rows(7));
        
    }

    std::cout << "Load learning model and test policy ... ";
    //load koopman operator
    if(true){
        Eigen::MatrixXd G_operator = robot.loadTxtData(filePath + "online_koopman_data/without_active/G_operator.csv", 49, 49);
        Eigen::MatrixXd A_operator = robot.loadTxtData(filePath + "online_koopman_data/without_active/A_operator.csv", 49, 49);
        arma::mat armaMat_a(A_operator.data(), A_operator.rows(), A_operator.cols(), false, true);
        arma::mat armaMat_g(G_operator.data(), G_operator.rows(), G_operator.cols(), false, true);
        robot.ksys->loadOperator(armaMat_a, armaMat_g);

        // Home to initial joint pose
        // target_ee_state = { 0.0, -0.54, 0.62, 1.0, 0.0, 0.0, 0.0 };
        target_ee_state = { 0.554, 0.0, 0.62, 1.0, 0.0, 0.0, 0.0 };
        target_joint_state = robot.solveIK(target_ee_state);
        robot.controller->obj->xd.head_rows(7) = target_joint_state;
        std::cout << target_joint_state << std::endl;
        arma::vec error = {0,0,0, 0,0,0,0};
        std::vector<arma::vec> error_push;
        arma::mat error_data;

        //joint_state_collector.clear();
        torque_controller_collector.clear();
        _tcurr = 0;
        while (ros::ok() && _tcurr < 50) {
            
            cdataIn = robot.controller->get_control(dataIn);
            command_torque = filter_value * command_torque + (1-filter_value)*cdataIn;
            safty_torques = robot.checkJointLimits();
            command_torque += safty_torques; 
            //command_torque=tor; //test

            robot.forwardTorqueCommandState(command_torque);

            loop_rate.sleep();
            _tcurr += _dt;
            ros::spinOnce();
            dataOut = robot.getDataState();
            robot.ksys->gradStep(dataIn, cdataIn, dataOut);
            dataIn = dataOut;
           
            joint_state_collector.push_back(robot.getDataState());
            u_collector.push_back(command_torque);
            end_pose_collector.push_back(robot.endEffector_pose);
            xd_collector.push_back(robot.controller->obj->xd.head_rows(7));
          
        }
    }
    
    robot.endCommandState(command_torque);
    std::cout << "Policy test end ... \n";
   
    std::string data_type = "online_koopman_data/";
    int num = 2;

    joint_state_data.resize(joint_state_collector[0].n_rows, joint_state_collector.size());
    for (int i = 0; i < joint_state_collector.size(); i ++) {
        joint_state_data.col(i) = joint_state_collector[i];
    }
    joint_state_data.save( filePath + data_type + "joint_state_data_" + std::to_string(num) + ".csv", arma::raw_ascii);

    end_pose_data.clear();
    end_pose_data.resize(end_pose_collector[0].n_rows, end_pose_collector.size());
    for (int i = 0; i < end_pose_collector.size(); i ++) {
        end_pose_data.col(i) = end_pose_collector[i];
    }
    end_pose_data.save( filePath + data_type + "end_pose_data_" + std::to_string(num) + ".csv", arma::raw_ascii);
    
    u_data.clear();
    u_data.resize(u_collector[0].n_rows, u_collector.size());
    for (int i = 0; i < u_collector.size(); i ++) {
        u_data.col(i) = u_collector[i];
    }
    u_data.save( filePath + data_type + "u_data_" + std::to_string(num) + ".csv", arma::raw_ascii);
    robot.ksys->saveOperator(filePath + "online_koopman_data/");
    
    ROS_INFO("Koopman Model Save successfully ...");
    return 0;
}