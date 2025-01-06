#!/usr/bin/env python
import numpy as np
from numpy.linalg import inv
import rospy
from sawyer_data_torque_control.srv import LQRController, LQRControllerResponse
from scipy.signal import place_poles
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are, solve_continuous_are
import time

class LQR():
    def __init__(self, A, B, Q, R) -> None:
        self.C = np.diag(np.ones(A.shape[0]))
        self.D = np.diag(np.ones(A.shape[0]))
        desired_poles = [0.5]*A.shape[0]
        
        # place_result = place_poles(A.T, self.C.T, desired_poles)
        # self.L = place_result.gain_matrix.T
        self.Bd = np.diag(np.ones(14)) 
        self.Dd = np.diag(np.ones(14)) 
        self.L = np.diag(np.ones(14)*100) 
        
        self.x = np.zeros((14,1))
        
        self.hat_x = np.zeros((A.shape[0], 1)) 
        self.u = np.zeros((B.shape[1], 1)) 
        self.hat_z = np.zeros((14,1))
        self.hat_d = np.zeros((14,1))
        
        self.hat_d_data = []
        self.hat_u = []  # kdx* hatd
        self.real_error_x = []
        self.input_data = []
        self.u_data = []
        self.remove_hat_u = []
        self.error_u_hat = []
        self.hat_z_data = []
        self.hat_y = []
        self.u_filter = []
        
        self.A_data = []
        self.B_data = []
        
        self.u_disturbance = np.zeros((7,1))
        self.u_disturbance[3] = 5
        self.count = 0
        
        self.Klqr = self.compute_LQR_gain(A, B, Q, R)
        self.Kdx = -np.linalg.pinv(B) @ self.Bd
        self.Kdy = np.linalg.pinv(self.C @ np.linalg.inv(A + B @ self.Klqr) @ B) @ self.Dd

    def compute_K(self, A, B, Q, R):
        # if np.linalg.norm(B, ord=2)<0.6:
        self.Klqr = self.compute_LQR_gain(A, B, Q, R)
        self.Kdx = -np.linalg.pinv(B) @ self.Bd
        self.Kdy = np.linalg.pinv(self.C @ np.linalg.inv(A + B @ self.Klqr) @ B) @ self.Dd
    
    def compute_LQR_gain(self, _A, _B, _Q, _R, maxIter=100, eps=1e-7):
        A = _A
        B = _B
        Q = _Q
        R = _R
        P = Q

        # Pold = P
        # for i in range(maxIter):
        #     # 计算 Riccati 方程的下一步迭代
        #     P = A.T @ P @ A - (A.T @ P @ B) @ inv(R + B.T @ P @ B) @ (B.T @ P @ A) + Q
        #     # 计算变化量以检查收敛性
        #     delta = Pold - P
        #     if np.abs(delta).max() < eps:
        #         break
        #     Pold = P

        P = solve_discrete_are(A, B, Q, R)
        # P = solve_continuous_are(A, B, Q, R)
        
        # 计算 LQR 增益
        Klqr = inv(R + B.T @ P @ B) @ (B.T @ P @ A)
        return Klqr

    def get_control(self, A, B, Q, R, x, xd):
        # self.hat_x = A @ self.hat_x + B @ self.u + self.L @ (x - self.C @ self.hat_x)
        # 
        # self.compute_K(A, B, Q, R)
        self.A_data.append(np.linalg.norm(A, ord=2))
        self.B_data.append(np.linalg.norm(B, ord=2))
        
        dt = 0.01
        
        dot_z = - self.L @ self.Bd @ (self.hat_z + self.L @ x) - self.L @ ((A-np.eye(14))/dt @ x + B/dt @ self.u)
        k1 = dot_z
        k2 = dot_z + dt * k1 / 2
        k3 = dot_z + dt * k2 / 2
        k4 = dot_z + dt * k3
        self.hat_z = self.hat_z + dt * (k1 + 2*k2 +2*k3 +k4)/6  
        hat_d = self.hat_z + self.L @ x
        
        # # print(hat_d.shape)
        # alfa = 0.5  # 0.1
        # hat_d = alfa * hat_d + (1-alfa) * self.hat_d
        
        # good paramter: 0.05 , 0.06, 0.08
        # good1: 100-0.06
        # good2: 100-0.06
        kk = 0.007  # good paramter: 0.06
        self.hat_z_data.append(self.hat_z[3,:])
        self.hat_d_data.append(hat_d[3,:])
        self.hat_u.append(kk * (self.Kdx @ hat_d)[3,:])
        self.real_error_x.append(((x - A @ self.x - B @ self.u))[3,:])
        self.error_u_hat.append(x[3,:])
        self.hat_y.append((kk*self.Kdx @ hat_d + 0.08*self.Kdy @ (x - A @ self.x - B @ self.u))[0,:])
        
        # np.zeros((7,1))
        # u = -3*self.Klqr[:, :7] @ (x[:7,:] - xd[:7,:]) - 1.0*self.Klqr[:, 7:] @ (x[7:,:] - xd[7:,:]) + kk*self.Kdx @ hat_d - 0.03*self.Kdy @ (x - A @ self.x - B @ self.u)
        u = -self.Klqr @ (x - xd) #+ kk*self.Kdx @ self.hat_d #- 0.03*self.Kdy @ (x - A @ self.x - B @ self.u) 
        # u = -self.Klqr @ (x - xd) 
        # u = -self.Klqr @ (x - xd) + kk*self.Kdx @ hat_d - 0.03*self.Kdy @ (x - A @ self.x - B @ self.u)
        # u = -3*self.Klqr[:, :7] @ (x[:7,:] - xd[:7,:]) - 1.0*self.Klqr[:, 7:] @ (x[7:,:] - np.zeros((7,1)))
        # self.hat_d = hat_d
        self.u = u
        # self.x = x
        
        # if self.count>1000 :
        #     u = u + self.u_disturbance * 1
        #     print("input disturbance ... ")
        
        self.count += 1
        self.remove_hat_u.append(((-self.Klqr @ (x - xd))[3,:] + kk * (self.Kdx @ hat_d))[3,:])
        self.u_data.append((-self.Klqr @ (x - xd))[3,:])
        
        # u = -20*self.L @ (x - xd) - 1 * hat_d 
        # k = 0.1
        # u = k * self.u + (1-k)*u
        
        return u
    
    def exponential_moving_average(self, data, alpha):
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema

def handle_greeting(req):
    x = np.array(req.joint_state)[:, np.newaxis]
    xd = np.array(req.xd)[:, np.newaxis]

    A_ = np.array(req.A).reshape(14, 14).T
    B_ = np.array(req.B).reshape(7, 14).T
    # print(A_.shape)
    start_t = time.time()
    torque = lqr.get_control(A_, B_, Q, R, x, xd)
    
    lqr.input_data.append(np.array(req.joint_state + req.xd))
    # lqr.u_data.append(torque.reshape(-1)[0])
    
    response = LQRControllerResponse()
    response.torque = tuple(torque)
    print("mpc controller runing time: ", time.time() - start_t)
    return response

if __name__ == "__main__":
    filePath = '/home/master/milab/franka_ws/src/sawyer_data_torque_control/data/'
    A = np.loadtxt(filePath + 'online_koopman_data/Alin.csv')
    B = np.loadtxt(filePath + 'online_koopman_data/Blin.csv')
    Q = np.loadtxt(filePath + 'online_koopman_data/Q.csv')
    R = np.loadtxt(filePath + 'online_koopman_data/R.csv')
    
    lqr = LQR(A, B, Q, R)
    rospy.init_node('online_learning')
    s = rospy.Service('online_learning_torque', LQRController, handle_greeting)
    print("Ready to greet.")
    
    rospy.spin()
    
    np.savetxt(filePath + "gp_data/real_error_x.csv", np.array(lqr.real_error_x), fmt='%f', delimiter=' ')
    np.savetxt(filePath + "gp_data/hat_d_data.csv", np.array(lqr.hat_d_data), fmt='%f', delimiter=' ')
    
    fig, ax = plt.subplots(4, 1, figsize=(15, 10))
    ax[0].plot(lqr.real_error_x, label='real_d')
    ax[0].plot(lqr.hat_d_data, label='hat_d')
    # ax[0].plot(lqr.hat_y, label='hat_y')
    ax[0].legend()
    
    # ax[1].plot(lqr.hat_u, label='hat_u')
    ax[1].plot(lqr.hat_d_data, label='hat_d')
    ax[1].plot(lqr.error_u_hat, label='x')  
    # ax[1].plot(lqr.hat_z_data, label='hat_z')
    ax[1].legend()
    
    ax[2].plot(lqr.u_data, label='u_data')
    ax[2].plot(lqr.remove_hat_u, label='remove_hat_u')
    ax[2].plot(lqr.hat_u, label='hat_u')
    # ax[2].plot(lqr.hat_y, label='hat_y',color='k')
    ax[2].legend()
    
    ax[3].plot(lqr.A_data, label='A_data')
    ax[3].plot(lqr.B_data, label='B_data')  
    ax[3].legend()

    plt.tight_layout()
    plt.legend()
    plt.show()