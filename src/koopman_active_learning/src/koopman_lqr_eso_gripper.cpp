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

#include "robotlib/eso.hpp"
#include "robotlib/task_traj.hpp"

#include "sawyer_data_torque_control/Dobdata.h"

#include <iostream>
#include <chrono>
#include <thread>

int main(int argc, char** argv) {
    ros::init(argc, argv, "lqr_dob_filter");

    float filter_value = 0.5;
    ros::NodeHandle nh;
    ros::Rate loop_rate(100);
    float _tcurr = 0.0;
    float _dt = 1.0/100.0;

    SawyerRobot robot(&loop_rate, _dt);

    std::cout<<"robot init successful"<<std::endl;
    
    arma::vec state = robot.getDataState();
    arma::vec target_ee_state = { 0.6, 0.0, 0.3, 0.5, 0.5, 0.5, 0.5 };
    arma::vec target_joint_state = robot.getIKState(target_ee_state);
    arma::vec homeing_joint_state = {0,0,0, 0,0,0,0};

    //save data varible
    std::vector<arma::vec> joint_state_collector;  // joint angle and val
    std::vector<arma::vec> torque_lqr_collector;
    std::vector<arma::vec> torque_eso_collector;
    std::vector<arma::vec> torque_distrubance_collector;
    std::vector<arma::vec> torque_input_collector;

    std::vector<arma::vec> pose_error_collector;  // end pos error
    std::vector<arma::vec> end_pose_collector;    // end pose
    std::vector<arma::vec> end_val_collector;     // end val
    std::vector<arma::vec> des_pose_collector;    // des pose
    std::vector<arma::vec> des_val_collector;
    

    std::string filePath = "/home/master/milab/franka_ws/src/sawyer_data_torque_control/data/";
    arma::mat joint_state_data;
    arma::mat torque_lqr_data;
    arma::mat torque_eso_data;
    arma::mat torque_distrubance_data;
    arma::mat torque_input_data;

    arma::mat pose_error_data;
    arma::mat end_pose_data;
    arma::mat end_val_data;
    arma::mat des_pose_data;
    arma::mat des_val_data;

    arma::mat jacobian;

    arma::vec command_torque = arma::zeros<arma::vec>(7);

    // ----------- System ID ----------- //
    std::cout << "Active Learning dynamic ... \n";
    float timer = 0;
    arma::vec safty_torques = arma::zeros<arma::vec>(7);
    arma::vec cdataIn = arma::zeros<arma::vec>(7);
    arma::vec dataIn = robot.getDataState();
    arma::vec dataOut;
    
    // ----------------- Home to zero ----------------------------- //
    robot.controller->umax[1] = 10;
    // robot.controller->umax[5] = 10;
    // robot.controller->umax[4] = 10;
    robot.controller->obj->q_fisher = 0;
    robot.controller->obj->xd.head_rows(7) = homeing_joint_state;
    robot.switchWeights();
    
    // Home to initial joint pose
    std::cout << "Initial pose ... \n";
    //load koopman operator
    Eigen::MatrixXd G_operator = robot.loadTxtData(filePath + "online_koopman_data/G_operator.csv", 49, 49);
    Eigen::MatrixXd A_operator = robot.loadTxtData(filePath + "online_koopman_data/A_operator.csv", 49, 49);
    arma::mat armaMat_a(A_operator.data(), A_operator.rows(), A_operator.cols(), false, true);
    arma::mat armaMat_g(G_operator.data(), G_operator.rows(), G_operator.cols(), false, true);
    robot.ksys->loadOperator(armaMat_a, armaMat_g);

    target_ee_state = { 0.544, 0.0, 0.65, 1.0, 0.0, 0.0, 0.0 };
    target_joint_state = robot.solveIK(target_ee_state);
    robot.controller->obj->xd.head_rows(7) = target_joint_state;
    std::cout << target_joint_state << std::endl;
    arma::vec error = {0,0,0, 0,0,0,0};
    std::vector<arma::vec> error_push;
    arma::mat error_data;

    while (ros::ok() && _tcurr < 5) {
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
        
    }
    robot.endCommandState(command_torque);
    // robot.ksys->saveOperator(filePath + "online_koopman_data/");
    
    
    // ----------------- Trajectory Tracking ----------------------------- //
    std::cout << "Trajectory Tracking ... \n";
    
    float omega = 2;
    float amp = 0.1;
    // robot.controller->umax[0] = 10;
    // robot.controller->umax[1] = 10;
    robot.switchWeights(false);
    // robot.controller->umax = 8*arma::ones<arma::vec>(7);

    arma::mat Alin = robot.ksys->fdx(robot.controller->obj->xd, arma::zeros<arma::vec>(7));
    arma::mat Blin = robot.ksys->fdu(robot.controller->obj->xd, arma::zeros<arma::vec>(7));
    arma::mat Qlqr = robot.controller->obj->Q;
    arma::mat Rlqr = robot.controller->obj->R;

    LQRController lqRegulator(Alin, Blin, Qlqr, Rlqr);
    ESO eso(dataOut, cdataIn);
    Trajectory traj;

    _tcurr = 0;
    arma::vec pose = arma::zeros<arma::vec>(7);
    arma::vec vel = arma::zeros<arma::vec>(6);
    arma::vec acc = arma::zeros<arma::vec>(6);
    arma::vec joint_state = arma::zeros<arma::vec>(7);
    arma::vec online_torque = arma::zeros<arma::vec>(7);
    jacobian = robot.getJacobian();

    arma::vec pre_hat_d_eso = arma::zeros<arma::vec>(7);
    float duration_t = 0.0;
    arma::vec dot_xd = arma::zeros<arma::vec>(14);

    //read gripper traj data
    Eigen::MatrixXd gripper_traj = robot.loadTxtData("/home/master/milab/franka_ws/src/sawyer_data_torque_control/data/task_traj/gripper_traj.csv", 3000, 3);
    
    arma::vec RAL_pre = arma::zeros<arma::vec>(3); 
    int index = 0;

    //initial pose at traj start point
    while( ros::ok() && _tcurr < 5) {
        pose = {0.5, 0.0, 0.5, -0.658, -0.262, -0.262, 0.656}; //[x, y, z, w]
        robot.controller->obj->xd.head_rows(7) = robot.solveIK(pose);

        Alin = robot.ksys->fdx(robot.controller->obj->xd, arma::zeros<arma::vec>(7));
        Blin = robot.ksys->fdu(robot.controller->obj->xd, arma::zeros<arma::vec>(7));

        // LQR controller
        lqRegulator.compute_LQR_gain(Alin, Blin, Qlqr, Rlqr);
        cdataIn = lqRegulator.getControl(dataOut, robot.controller->obj->xd) ;

        filter_value = 0.2;
        command_torque = filter_value * command_torque + (1-filter_value)*cdataIn;
        safty_torques = robot.checkJointLimits();
        command_torque += safty_torques; 

        robot.forwardTorqueCommandState(command_torque);
        _tcurr += _dt;        
        
        loop_rate.sleep(); 
        ros::spinOnce(); 
        dataOut = robot.getDataState();
        robot.ksys->gradStep(dataIn, cdataIn, dataOut);

        dataIn = dataOut;
    }

    _tcurr = 0;
    //#############################################
    bool method =  false      ;   // false:baseline true:proposed
    //#############################################
    while( ros::ok() && _tcurr < 50) {
        auto start = std::chrono::high_resolution_clock::now();
        // gripping trajectory
        if(_tcurr < 10){
            pose[0] = gripper_traj(index, 0);
            pose[1] = gripper_traj(index, 1);
            pose[2] = gripper_traj(index, 2);
            // pose[3] = 1;
            pose[3] = -0.658;
            pose[4] = -0.262;
            pose[5] = -0.262;
            pose[6] = 0.656;

            index += 1;
            std::cout << "Step1: go to bottle ..." << std::endl;
        } 
        
        else if(_tcurr >= 10 && index == 1000){
            robot.GripperCommandGet();
            index += 1;
            std::cout << "Step2: gripping ..." << std::endl;
        }
        
        else if(_tcurr > 15 && index < 3000){
            pose[0] = gripper_traj(index, 0);
            pose[1] = gripper_traj(index, 1);
            pose[2] = gripper_traj(index, 2);
            // pose[3] = 1;
            pose[3] = -0.658;
            pose[4] = -0.262;
            pose[5] = -0.262;
            pose[6] = 0.656;

            index += 1;
            std::cout << "Step3: gripping bottle to goal ..." << std::endl;
        }
        
        else if(index == 3000){
            robot.GripperCommandStop();
            index += 1;
            std::cout << "Step4: put ..." << std::endl;
        }
        
        joint_state = robot.solveIK(pose);
        robot.controller->obj->xd.head_rows(7) = joint_state;

        arma::vec pose_error = arma::zeros<arma::vec>(6);
        for(int i=0; i<3; i++){
            pose_error[i] = robot.endEffector_pose[i] - pose[i];
        }
        arma::vec dqr = arma::pinv(jacobian) * vel ;
        robot.controller->obj->xd.tail_rows(7) = dqr;
       
        // get dynamic matrix
        Alin = robot.ksys->fdx(robot.controller->obj->xd, arma::zeros<arma::vec>(7));
        Blin = robot.ksys->fdu(robot.controller->obj->xd, arma::zeros<arma::vec>(7));
        
        end_pose_collector.push_back(robot.endEffector_pose);
        des_pose_collector.push_back(pose);
        pose_error_collector.push_back(pose_error);
        joint_state_collector.push_back(dataOut);

        // ESO estimater
        arma::vec hat_d_eso = eso.estimate_hat(dataIn);

        // LQR controller
        lqRegulator.compute_LQR_gain(Alin, Blin, Qlqr, Rlqr);
        arma::vec lqr_u = lqRegulator.getControl(dataOut, robot.controller->obj->xd) ;//+ ddq;
        if (method){ 
            cdataIn = lqr_u - 0.1*hat_d_eso;  //proposed
        }
        else{
            cdataIn = lqr_u;  //baseline
        }
        
        eso.get_u(cdataIn);

        torque_lqr_collector.push_back(lqr_u);
        torque_eso_collector.push_back(0.1*hat_d_eso);
        torque_input_collector.push_back(cdataIn);

        //add constant distrubance in the end of arm
        arma::vec end_f_distrubance = arma::zeros(6);
        arma::vec u_distrubance = arma::zeros(7);
        if (_tcurr>10){
            end_f_distrubance[0] = 0;  //x
            end_f_distrubance[1] = 0;  //y
            end_f_distrubance[2] = 0;  //z
            u_distrubance = jacobian.t()*end_f_distrubance;
            cdataIn = cdataIn + u_distrubance;
        }
        torque_distrubance_collector.push_back(u_distrubance);

        
        filter_value = 0.2;
        command_torque = filter_value * command_torque + (1-filter_value)*cdataIn;
        safty_torques = robot.checkJointLimits();
        command_torque += safty_torques; 

        robot.forwardTorqueCommandState(command_torque);
    
        _tcurr += _dt;        
        
        loop_rate.sleep(); 
        ros::spinOnce(); 
        
        dataOut = robot.getDataState();
        robot.ksys->gradStep(dataIn, cdataIn, dataOut);

        dataIn = dataOut;

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(end - start);
        duration_t = duration.count();
        // std::cout << "Execution time: " << duration.count() << " milliseconds" << std::endl;

    }
    // save data file_path
    // std::string data_type = "eso_koopman_data/constant_distrubance/";
    // std::string data_type = "eso_koopman_data/dynamic_distrubance/";
    // std::string data_type = "eso_koopman_data/LQR_overshoot/";
    // std::string data_type = "eso_koopman_data/change_task/";
    std::string data_type = "eso_koopman_data/without_distrubance/";

    //------------------------------------1
    // num=1 baseline(False)  num=2 propose(True)
    int num;
    if(method){
        num = 2;
    }
    else{
        num = 1;
    }
    
    pose_error_data.clear();
    pose_error_data.resize(pose_error_collector[0].n_rows, pose_error_collector.size());
    for (int i = 0; i < pose_error_collector.size(); i ++) {
        pose_error_data.col(i) = pose_error_collector[i];
    }
    pose_error_data.save( filePath + data_type + "pose_error_data_" + std::to_string(num) + ".csv", arma::raw_ascii);
    //------------------------------------2
    end_pose_data.clear();
    end_pose_data.resize(end_pose_collector[0].n_rows, end_pose_collector.size());
    for (int i = 0; i < end_pose_collector.size(); i ++) {
        end_pose_data.col(i) = end_pose_collector[i];
    }
    end_pose_data.save( filePath + data_type + "end_pose_data_" + std::to_string(num) + ".csv", arma::raw_ascii);
    //------------------------------------3
    des_pose_data.clear();
    des_pose_data.resize(des_pose_collector[0].n_rows, des_pose_collector.size());
    for (int i = 0; i < des_pose_collector.size(); i ++) {
        des_pose_data.col(i) = des_pose_collector[i];
    }
    des_pose_data.save( filePath + data_type + "des_pose_data_" + std::to_string(num) + ".csv", arma::raw_ascii);
    //------------------------------------4
    joint_state_data.clear();
    joint_state_data.resize(joint_state_collector[0].n_rows, joint_state_collector.size());
    for (int i = 0; i < joint_state_collector.size(); i ++) {
        joint_state_data.col(i) = joint_state_collector[i];
    }
    joint_state_data.save( filePath + data_type + "joint_state_data_" + std::to_string(num) + ".csv", arma::raw_ascii);
    //------------------------------------5
    torque_lqr_data.clear();
    torque_lqr_data.resize(torque_lqr_collector[0].n_rows, torque_lqr_collector.size());
    for (int i = 0; i < torque_lqr_collector.size(); i ++) {
        torque_lqr_data.col(i) = torque_lqr_collector[i];
    }
    torque_lqr_data.save( filePath + data_type + "torque_lqr_data_" + std::to_string(num) + ".csv", arma::raw_ascii);
    //------------------------------------6
    torque_eso_data.clear();
    torque_eso_data.resize(torque_eso_collector[0].n_rows, torque_eso_collector.size());
    for (int i = 0; i < torque_eso_collector.size(); i ++) {
        torque_eso_data.col(i) = torque_eso_collector[i];
    }
    torque_eso_data.save( filePath + data_type + "torque_eso_data_" + std::to_string(num) + ".csv", arma::raw_ascii);
    //------------------------------------7
    torque_input_data.clear();
    torque_input_data.resize(torque_input_collector[0].n_rows, torque_input_collector.size());
    for (int i = 0; i < torque_input_collector.size(); i ++) {
        torque_input_data.col(i) = torque_input_collector[i];
    }
    torque_input_data.save( filePath + data_type + "torque_input_data_" + std::to_string(num) + ".csv", arma::raw_ascii);

    //------------------------------------8
    torque_distrubance_data.clear();
    torque_distrubance_data.resize(torque_distrubance_collector[0].n_rows, torque_distrubance_collector.size());
    for (int i = 0; i < torque_distrubance_collector.size(); i ++) {
        torque_distrubance_data.col(i) = torque_distrubance_collector[i];
    }
    torque_distrubance_data.save( filePath + data_type + "torque_distrubance_data_" + std::to_string(num) + ".csv", arma::raw_ascii);
   

    return 0;
}