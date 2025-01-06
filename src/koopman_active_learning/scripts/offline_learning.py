import numpy as np
import math
import matplotlib.pyplot as plt
import random
from scipy import signal
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
import time
from multiprocessing import shared_memory
import concurrent.futures


class Off_Network:
    def __init__(self, input_size, hidden_size, output_size) -> None:
        self.input_num = input_size
        self.hidden_num = hidden_size
        self.output_num = output_size
        self.parallel_num = 6
        
        # init weight
        self.w1 = np.zeros((self.hidden_num,))
        self.w2 = np.zeros((output_size,self.hidden_num)) # 6x7x9
        for i in range(self.hidden_num):
            self.w1[i]= 1 * (random.uniform(0.2, 0.9))
                
        # self.K = 0.2
        # self.Lo = 0.9
        self.K = 0.9
        self.Lo = 0.06
        
        self.start_time = 0.
        self.end_time = 0.
        self.loss_histry = []
        
        self.histry_train_pose_data = []
        self.histry_q_pose_data = []
        self.gradent = []
        
        self.config_init = 0
        
    # ======================================   工具函数   =============================================   
    def save_data(self, txt_name, data):
        # 保存为文本文件
        # print(data.shape)
        np.savetxt(txt_name, data.reshape(data.shape[0], -1), fmt='%lf', delimiter='\t')
    
    def read_data(self, file_name):
        loaded_data = np.loadtxt(file_name, dtype=float, delimiter='\t')
        return loaded_data
    
    def filter_data(self, data_size, q_data, end_pos_data, dot_t=0.013, sigma=2, flag=False):
        RobotPos1 = end_pos_data[-data_size:,]
        JointPos1 = q_data[-data_size:,]
        
        RobotVel = np.zeros(RobotPos1.shape)
        JointVel = np.zeros(JointPos1.shape)

        # gaussian fiter
        if flag == True:
            for i in range(RobotPos1.shape[1]):
                RobotPos1[:,i] = gaussian_filter1d(RobotPos1[:, i], sigma)
            for i in range(JointPos1.shape[1]):
                JointPos1[:,i] = gaussian_filter1d(JointPos1[:, i], sigma)

        # get vel
        for i in range(1,RobotPos1.shape[0]):
            RobotVel[i,]=(RobotPos1[i,]-RobotPos1[i-1,]) / dot_t
            JointVel[i,] = (JointPos1[i,] - JointPos1[i - 1,]) / dot_t
        RobotVel[0,]=RobotVel[2,]
        JointVel[0,]=JointVel[2,]
        
        return RobotPos1, JointPos1, RobotVel, JointVel
    # ================================================================================================
    
    # ===========================  train offline model by collecting data  ===========================
    def forward(self, q_data):
        # neuron1 = 9
        self.theta=np.zeros((q_data.shape[0],self.hidden_num))
        
        # 初始化 theta
        for i in range(q_data.shape[0]):
            for j in range(self.hidden_num):
                Jq=q_data[i,j]
                aa=self.w1[j]
                self.theta[i,j]=1/(1+math.exp(-aa*Jq))
                
    def Jacobian_learning(self, des_torque, w2): 
        error_sum = np.zeros((des_torque.shape[1], ))
    
        Est_tor = np.zeros(des_torque.shape)
        Est_error = np.zeros(des_torque.shape)
       
        for i in range(des_torque.shape[0]):
            Est_torque = np.dot(w2, self.theta[i,]) + self.K * error_sum
            torque_error = des_torque[i,] - Est_torque
            error_sum = error_sum + torque_error
            theta_qd = self.theta[i, :]  # 7x9
            gradent_off = self.Lo * theta_qd[np.newaxis, :] * error_sum[:, np.newaxis]
            w2 = w2 + gradent_off
            
            self.gradent.append(np.std(gradent_off))
           
            Est_tor[i,] = Est_torque
            Est_error[i,] = error_sum
        return Est_tor, Est_error, w2
    
    def train(self, joint_state, des_torque, epochs=800):
        # update theta
        self.forward(joint_state)
        pbar = tqdm(range(epochs))
        for i in pbar:
            (Est_v, Est_error, W2) = self.Jacobian_learning(des_torque, self.w2)
            self.w2 = W2
            Loss = np.std(Est_error)
            self.loss_histry.append(Loss)
            pbar.set_description(f'Epoch {i} Loss {Loss}')
        # return Loss
            
    def test_model(self, joint_state, des_torque):
        self.forward(joint_state)
        
        self.LearnTorque = np.zeros(des_torque.shape)
        for i in range(des_torque.shape[0]):
            self.LearnTorque[i,] = np.dot(self.w2, self.theta[i,])
        
        print("start plot")
        # fig = plt.figure(figsize=(5,10))
        # ax = fig.add_subplot(511, projection='3d')
        # plt.plot(RobotPos_[:,0],RobotPos_[:,1],RobotPos_[:,2], '--')
        # plt.plot(predict_pos[:,0], predict_pos[:,1], predict_pos[:,2])
        # plt.ylabel('est_pos axis & end_pos')
        
        plt.subplot(811)
        plt.plot(des_torque[:,0],'--')
        plt.plot(self.LearnTorque[:,0])
        plt.ylabel('joint1 torque')
        
        plt.subplot(812)
        plt.plot(des_torque[:,1],'--')
        plt.plot(self.LearnTorque[:,1])
        plt.ylabel('joint2 torque')
        
        plt.subplot(813)
        plt.plot(des_torque[:,2],'--')
        plt.plot(self.LearnTorque[:,2])
        plt.ylabel('joint3 torque')
        
        plt.subplot(814)
        plt.plot(des_torque[:,3],'--')
        plt.plot(self.LearnTorque[:,3])
        plt.ylabel('joint4 torque')
        
        plt.subplot(815)
        plt.plot(des_torque[:,4],'--')
        plt.plot(self.LearnTorque[:,4])
        plt.ylabel('joint4 torque')
        
        plt.subplot(816)
        plt.plot(des_torque[:,5],'--')
        plt.plot(self.LearnTorque[:,5])
        plt.ylabel('joint4 torque')
        
        plt.subplot(817)
        plt.plot(des_torque[:,6],'--')
        plt.plot(self.LearnTorque[:,6])
        plt.ylabel('joint4 torque')
        
        plt.subplot(818)
        plt.plot(self.loss_histry)
        plt.ylabel('Loss')
        plt.tight_layout()

        plt.show()
        print("end plot")
            
    def run_offline_train(self):
        # load data 
        filePath = '/home/master/milab/franka_ws/src/sawyer_data_torque_control/data/'
        # JointState = np.loadtxt(filePath + 'offline_train_data/joint_state_data.csv').T  
        # Torque = np.loadtxt(filePath + 'offline_train_data/torque_data.csv').T      # 关节角度， mx1x7， m=数据数量
        JointState = np.loadtxt(filePath + 'franka_8_trajectory_data/joint_positions.csv')[:,:7] 
        JointState = np.hstack((JointState, np.loadtxt(filePath + 'franka_8_trajectory_data/joint_velocities.csv')[:,:7]))
        Torque = np.loadtxt(filePath + 'franka_8_trajectory_data/joint_efforts.csv')[:,:7]       # 关节角度， mx1x7， m=数据数量
        
        print(JointState.shape)
        data_size = 6000
        # RobotPos1, JointPos1, RobotVel, JointVel = self.filter_data(data_size, JointPos, RobotPos, flag=True)

        # train
        self.train(JointState, Torque, epochs=1000)
        
        # for i in range(4):
        #     self.train(JointPos1[1500*i:1500*(i+1),:], JointVel[1500*i:1500*(i+1),:], RobotVel[1500*i:1500*(i+1),:], epochs=1000)
            # self.loss_histry.append(Loss)
        
        # plt.plot(self.gradent)
        # plt.show()
        
        # test model
        self.test_model(JointState, Torque)
    # ================================================================================================
    

if __name__ == "__main__":
    nn = Off_Network(14, 14, 7)
    nn.run_offline_train()
    
    filePath = '/home/master/milab/franka_ws/src/sawyer_data_torque_control/data/'
    nn.save_data(filePath+'offline_train_data/weight_data/weight1_data.txt', nn.w1)
    nn.save_data(filePath+'offline_train_data/weight_data/weight2_data.txt', nn.w2)