#!/usr/bin/env python
import numpy as np
import math
import random
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
import time
import matplotlib.pyplot as plt
import time

import rospy
from sawyer_data_torque_control.srv import Koopman2Online, Koopman2OnlineResponse

class On_Network:
    def __init__(self, input_size=21, hidden_size=21, output_size=7) -> None:
        self.input_num = input_size
        self.hidden_num = hidden_size
        self.output_num = output_size
        
        # init weight
        self.w1 = np.zeros((self.hidden_num,))  # 14
        self.w2 = np.zeros((self.hidden_num, self.output_num)) # 14x7
        
        # self.w2 = np.random.randn(self.hidden_num, self.output_num)
        
        for i in range(self.hidden_num):
            self.w1[i]= 1 * (random.uniform(0.4, 0.6))
            
        self.theta=np.zeros((1, self.hidden_num))  # 1x14
            
        self.Lo = 0.2  # TODO
        
        self.start_time = 0.
        self.end_time = 0.
        self.loss_histry = []
        
    # ======================================   工具函数   =============================================   
    def save_data(self, txt_name, data):
        # 保存为文本文件
        # print(data.shape)
        np.savetxt(txt_name, data.reshape(data.shape[0], -1), fmt='%lf', delimiter='\t')
    
    def read_data(self, file_name):
        loaded_data = np.loadtxt(file_name, dtype=float, delimiter='\t')
        return loaded_data
    
    # ================================================================================================
    
    # =============================== online train by real data ======================================
    def forward(self, q_data):
        for j in range(self.hidden_num):
            Jq=q_data[j]
            # print(Jq.shape)
            aa=self.w1[j]
            # print(aa.shape)
            self.theta[0, j] = 1/(1+math.exp(-aa*Jq))
            
    def forward_full(self, q_data):
        for j in range(self.hidden_num):
            Jq=q_data[j]
            # print(Jq.shape)
            aa=self.w1[j]
            # print(aa.shape)
            self.theta[0, j] = 1/(1+math.exp(-aa*Jq))
            
    def Jacobian_learning(self, pose_error):
        # theta_qd = np.dot(qd_data.reshape(qd_data.shape[0],1), self.theta[0,].reshape(self.theta.shape[1], 1).T)  # 7x7
        gradent_on = self.Lo * np.dot(self.theta.T, pose_error)  # pose_error:1x7
        self.w2 = self.w2 - gradent_on
        
        return self.w2
    
    def online_train(self, q_data, pos_error, epochs=800):
        self.forward(q_data)
        # pbar = tqdm(range(epochs))v
        for i in range(epochs):
            w2 = self.Jacobian_learning(pos_error)
        
        return w2 
    
    def run_online(self, input_data, track_error, epochs=1):
        w2 = self.online_train(input_data, track_error, epochs)
        est_torque = np.dot(self.theta, w2)  # 1x7
        
        # low pass TODO
        # a = 0.001
        # est_torque = a/(est_torque + a)
        
        return est_torque
    
    
    # ================================================================================================
joint_state = np.zeros(14)
qr = np.zeros(7)


def handle_greeting(req):
    global count
     
    joint_state = np.array(req.joint_state)
    qr =  np.array(req.qr)

    input_data = np.hstack((joint_state, qr))
    track_error = joint_state[7:].reshape(1, 7) - qr.reshape(1, 7)
    # track_error = np.array(req.error).reshape(1, 7)
    
    tor = nn.run_online(input_data, track_error).reshape(-1)
    # print("torque:", tor)
    response = Koopman2OnlineResponse()
    response.torque = tuple(tor)
    print("response ... ")
    return response
  

if __name__ == "__main__":
    nn = On_Network(21, 21, 7)
    count = 0
    rospy.init_node('online_learning')
    s = rospy.Service('online_learning_torque', Koopman2Online, handle_greeting)
    print("Ready to greet.")
    
    rospy.spin()
   
    
    