import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import time

# load data
filePath = '/home/master/milab/franka_ws/src/sawyer_data_torque_control/data/'

des_pose_baseline = np.loadtxt(filePath + 'online_koopman_data/KPdata/des_pose_data_0.csv')
end_pose_lqr = np.loadtxt(filePath + 'online_koopman_data/KPdata/end_pose_data_1.csv')
end_pose_bod = np.loadtxt(filePath + 'online_koopman_data/KPdata/end_pose_data_3.csv')
end_pose_bod_p = np.loadtxt(filePath + 'online_koopman_data/KPdata/end_pose_data_4.csv')
# 创建一个初始图
plt.ion()  # 开启交互模式
fig, ax = plt.subplots()
line1, = ax.plot([], [], 'b-', linewidth=3, label='des_traj')
line2, = ax.plot([], [], 'r-', linewidth=3,label='lqr_traj')
line3, = ax.plot([], [], 'y-', linewidth=3,label='bod_traj')
line4, = ax.plot([], [], 'g-', linewidth=3,label='bod_P_traj')

display(fig)

# 初次绘制
fig.canvas.draw()
fig.canvas.flush_events()

x_des = []
y_des = []

x_bod = []
y_bod = []
x_bod_p = []
y_bod_p = []

x_lqr = []
y_lqr = []

end = 5000
start = 1000

for i in range(end-start):
    
    x_des.append(des_pose_baseline[1,start+i])
    y_des.append(des_pose_baseline[2,start+i])
    
    x_lqr.append(end_pose_lqr[1,start+i])
    y_lqr.append(end_pose_lqr[2,start+i])
    
    x_bod.append(end_pose_bod[1,start+i])
    y_bod.append(end_pose_bod[2,start+i])
    
    x_bod_p.append(end_pose_bod_p[1,start+i])
    y_bod_p.append(end_pose_bod_p[2,start+i])
    
    line1.set_xdata(x_des)
    line1.set_ydata(y_des)
    line2.set_xdata(x_lqr)
    line2.set_ydata(y_lqr)
    line3.set_xdata(x_bod)
    line3.set_ydata(y_bod)
    line4.set_xdata(x_bod_p)
    line4.set_ydata(y_bod_p)
    
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(0.3, 0.7)
    
    # 重新绘制
    ax.draw_artist(ax.patch)
    ax.draw_artist(line1)
    ax.draw_artist(line2)
    ax.draw_artist(line3)
    ax.draw_artist(line4)
    fig.canvas.blit(ax.bbox)
    fig.canvas.flush_events()

    # time.sleep(0.0001)  # 模拟延迟