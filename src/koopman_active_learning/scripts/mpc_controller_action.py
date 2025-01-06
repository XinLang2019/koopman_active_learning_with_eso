#!/usr/bin/env python
import numpy as np
from cvxopt import matrix, solvers
import time 
import rospy
import actionlib
from sawyer_data_torque_control.msg import MpcControlAction, MpcControlFeedback, MpcControlResult
from numpy.linalg import inv

class MPController():
    def __init__(self, A, B, Q, R, N):
        self.A = A
        self.B = B
        
        self.Q = Q
        self.R = R
        self.S = Q  # terminal cost
        self.N_P = N # horizon of mpc predict
        self._t = []
        self.phi, self.gamma, self.omega, self.psi= self.Create_QP_Matrix(A, B, Q, R, self.N_P)
        
        self.g = self.gamma.T @ self.omega
        self.H = self.psi + self.gamma.T @ self.omega @ self.gamma
        
    def Create_QP_Matrix(self, A, B, Q, R, N_P):
        '''
        Create matrices necessary for solving the quadratic programming (QP) problem in Model Predictive Control (MPC).
        
        '''

        # Calculate the dimensions of the system matrix A
        n = A.shape[0]
        # Calculate the dimensions of the input matrix B
        p = B.shape[1]
        
        # Initialize Phi matrix and define its dimensions
        Phi = np.zeros((N_P * n, n))
        # Initialize Gamma matrix and define its dimensions
        Gamma = np.zeros((N_P * n, N_P * p))
        
        # Define temporary identity matrix
        tmp = np.eye(n)
        # Define for-loop row vector
        rows = np.arange(n)
        
        # For loop to build Phi and Gamma matrices
        for i in range(N_P):
            # Build Phi matrix, refer to formula (5.3.5b)
            Phi[(i * n):((i + 1) * n), :] = np.linalg.matrix_power(A, i + 1)
            # Build Gamma matrix, refer to formula (5.3.5b)
            Gamma[rows, :] = np.hstack([tmp @ B, Gamma[np.maximum(0, rows - n), :-p]])
            # Update row number of Gamma matrix
            rows = (i+1) * n + np.arange(n)
            # Build temporary matrix for power calculations of matrix A
            tmp = A @ tmp
        
        # Build Omega matrix, including part of Q matrix
        Omega = np.kron(np.eye(N_P), Q)
        
        Psi = np.kron(np.eye(N_P), R)
        
        return Phi, Gamma, Omega, Psi
    
    def MPC_Controller_withConstraints(self, x, xd):
        '''
        Solve the quadratic programming problem for the MPC controller with constraints using cvxopt.
        
        '''
        p = self.B.shape[1]
        # Convert input matrices to cvxopt format
        # min x'Px + q'x
        # Gx <= h
        # Compute quadratic programming matrix F
        F = self.g @ (self.phi @ x - xd)
        
        print(F.shape)
        # print(F.shape)
        P = matrix(self.H)
        q = matrix(F)  # xd n-step reference trajectory
        
        # Solve the quadratic programming problem using cvxopt
        solvers.options['show_progress'] = False  # Optionally turn off solver output
        result = solvers.qp(P, q)

        # Extract the optimized control inputs over the prediction horizon
        U = np.array(result['x']).flatten()
       
        # Extract the first set of control inputs to apply
        u = U[:p]

        return U, u
    
_t = []
def handle_greeting(goal):
    x = np.array(goal.joint_state)[:, np.newaxis]
    xd = np.array(goal.xd)[:, np.newaxis]
    start = time.time()
    optimized_torque = mpc.MPC_Controller_withConstraints(x, xd)
    
    if _server.is_preempt_requested():
        rospy.loginfo('MPC Optimization was preempted')
        _server.set_preempted()
        return
    
    result = MpcControlResult()
    result.torque = tuple(optimized_torque)
    _server.set_succeeded(result)
    _t.append(time.time()-start)
    # print("mpc controller runing time: ", np.mean(_t))
    
    
if __name__ == "__main__":
    filePath = '/home/master/milab/franka_ws/src/sawyer_data_torque_control/data/'
    A = np.loadtxt(filePath + 'online_koopman_data/Alin.csv')
    B = np.loadtxt(filePath + 'online_koopman_data/Blin.csv')
    Q = np.loadtxt(filePath + 'online_koopman_data/Q.csv')
    R = np.loadtxt(filePath + 'online_koopman_data/R.csv')
    
    N_P = 20
    
    mpc = MPController(A, B, Q, R, N_P)
    
    rospy.init_node('mpc_controller')
    _server = actionlib.SimpleActionServer('mpc_control', MpcControlAction, execute_cb=handle_greeting, auto_start=False)
    _server.start()
    print("Ready to greet.")
    
    rospy.spin()
    
    
    