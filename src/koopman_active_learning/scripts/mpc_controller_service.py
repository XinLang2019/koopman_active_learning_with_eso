#!/usr/bin/env python
import numpy as np
from cvxopt import matrix, solvers
import time 
import rospy
from sawyer_data_torque_control.srv import MpcController, MpcControllerResponse
from numpy.linalg import inv
from scipy.signal import place_poles

class MPController():
    def __init__(self, A, B, Q, R, N):
        self.A = A
        self.B = B
        
        self.Q = Q
        self.R = R
        self.S = Q  # terminal cost
        self.N_P = N # horizon of mpc predict
        self.Qc = np.diag(np.ones(A.shape[0])*10)
        self.Rc = np.diag(np.ones(A.shape[0])*0.05)
        self.Hm = np.diag(np.ones(A.shape[0])*1.0)
        self.x_hat = np.array([0.0, 0.0, 0.0, -1.54, 0.0, 1.54, 0.0]+[0]*7)[:, np.newaxis]
        self.P = np.diag(np.ones(A.shape[0])*1.0)
        self.u = np.array([0]*7)[:, np.newaxis]
        
        self._t = []
        self.e_lkf = []
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
        # print(xd[:14,:].shape)
        # self.x_hat, _, self.P = self.LinearKalmanFilter(x, self.x_hat, self.P, self.u)
        # desired_poles = [0.5]*self.A.shape[0]
        # C = np.diag(np.ones(A.shape[0]))
        # place_result = place_poles(self.A.T, C.T, desired_poles)
        # self.L = place_result.gain_matrix.T
        self.phi, self.gamma, self.omega, self.psi= self.Create_QP_Matrix(A, B, Q, R, self.N_P)
        self.g = self.gamma.T @ self.omega
        self.H = self.psi + self.gamma.T @ self.omega @ self.gamma
        
        F = self.g @ (self.phi @ x  - xd)
        
        # print(self.H.shape)
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
        self.u = u[:,np.newaxis]
        
        return u
    
    import numpy as np

    def LinearKalmanFilter(self, z, x_hat, P, u):
        # 计算先验状态估计
        # print(self.A.shape)
        # print(P.shape)
        x_hat_minus = self.A @ x_hat + self.B @ u
        
        # 计算先验估计误差协方差矩阵
        P_minus = self.A @ P @ self.A.T + self.Qc
        
        # 计算卡尔曼增益
        K = P_minus @ self.Hm.T @ np.linalg.inv(self.Hm @ P_minus @ self.Hm.T + self.Rc)
        
        # 更新后验估计
        x_hat = x_hat_minus + K @ (z - self.Hm @ x_hat_minus)
        
        # 后验估计误差协方差矩阵
        P = (np.eye(self.A.shape[0]) - K @ self.Hm) @ P_minus
        
        self.e_lkf.append(x_hat - x_hat_minus)
        
        return x_hat, x_hat_minus, P

    
_t = []
def handle_greeting(req):
    x = np.array(req.joint_state)[:, np.newaxis]
    xd = np.array(req.xd)[:, np.newaxis]
    
    mpc.A = np.array(req.A).reshape(14, 14)
    mpc.B = np.array(req.B).reshape(14, 7)
    
    start = time.time()
    torque = mpc.MPC_Controller_withConstraints(x, xd)
    response = MpcControllerResponse()
    response.torque = tuple(torque)
    _t.append(time.time()-start)
    print("mpc controller runing time: ", np.std(_t, axis=0))
    return response
    
if __name__ == "__main__":
    filePath = '/home/master/milab/franka_ws/src/sawyer_data_torque_control/data/'
    A = np.loadtxt(filePath + 'online_koopman_data/Alin.csv')
    B = np.loadtxt(filePath + 'online_koopman_data/Blin.csv')
    Q = np.loadtxt(filePath + 'online_koopman_data/Q.csv')
    R = np.loadtxt(filePath + 'online_koopman_data/R.csv')
    
    N_P = 20
    
    mpc = MPController(A, B, Q, R, N_P)
    
    rospy.init_node('mpc_controller')
    s = rospy.Service('mpc_torque', MpcController, handle_greeting)
    print("Ready to greet.")
    
    rospy.spin()
    
    
    