#!/usr/bin/env python

import rospy
from sawyer_data_torque_control.msg import Dobdata
import matplotlib.pyplot as plt
import numpy as np

hat_d = []
input_u = []
lqr_u = []
state = []
def callback(msg):
    hat_d.append(1.*np.array(msg.hat_d))
    input_u.append(msg.input_u)
    lqr_u.append(msg.lqr_u)
    state.append(msg.state)
    print(np.array(hat_d).shape)

def listener():
    rospy.init_node('dob_plot', anonymous=True)
    rospy.Subscriber("dob_data", Dobdata, callback)
    print("Ready to greet.")
    rospy.spin()

if __name__ == '__main__':
    listener()
    index = 3
    fig, ax = plt.subplots(2, 1, figsize=(15, 10))
    ax[0].plot([np.mean(np.array(hat_d)[:,index])]*len(hat_d), label='mean_hat')
    ax[0].plot(np.array(hat_d)[:,index], label='hat_d')
    ax[0].legend()
    
    ax[1].plot(np.array(lqr_u)[:,index], label='lqr_u')
    ax[1].plot(np.array(hat_d)[:,index], label='hat_u')
    ax[1].plot(np.array(input_u)[:,index], label='input_u')
    ax[1].legend()
    
    fig, ax = plt.subplots(7, 1, figsize=(15, 10))
    for i in range(7):
        ax[i].plot(np.array(hat_d)[:,i], label='joint{}'.format(i+1))
        ax[i].plot([np.mean(np.array(hat_d)[:,i])]*len(np.array(hat_d)[:,i]), label='mean:{:.2f}'.format(np.mean(np.array(hat_d)[:,i])))

        ax[i].set_ylabel("hat_d")
        ax[i].legend(loc='lower right')
        
    fig, ax = plt.subplots(7, 1, figsize=(15, 10))
    for i in range(7):
        ax[i].plot(np.array(lqr_u)[:,i], label='lqr_u')
        ax[i].plot(np.array(hat_d)[:,i], label='hat_u')
        ax[i].plot(np.array(input_u)[:,i], label='input_u')

        ax[i].set_ylabel("hat_d")
        ax[i].legend(loc='lower right')
        
    fig, ax = plt.subplots(7, 1, figsize=(15, 10))
    for i in range(7):
        ax[i].plot(np.array(state)[:,i], label='joint_angle')
        ax[i].plot(np.array(hat_d)[:,i], label='hat_u')
        ax[i].plot(np.array(state)[:,i+7] - np.array(state)[:,i], label='error')

        ax[i].set_ylabel("hat_d")
        ax[i].legend(loc='lower right')
    
    plt.tight_layout()
    plt.show()
