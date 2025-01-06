#!/usr/bin/env python
import rospy
from intera_core_msgs.msg import JointCommand

def joint_torque_publisher():
    # 初始化ROS节点
    rospy.init_node('joint_torque_publisher')

    # 创建一个Publisher，发布到特定的topic
    pub = rospy.Publisher('/robot/limb/right/joint_command', JointCommand, queue_size=10)

    # 创建JointCommand消息实例
    torque_command = JointCommand()
    torque_command.mode = JointCommand.TORQUE_MODE  # 设置为力矩模式
    torque_command.names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
    torque_command.effort = [0.1] * 7  # 为所有7个关节设置相同的力矩值0.1

    # 设置发布频率
    rate = rospy.Rate(100)  # 10Hz
    while not rospy.is_shutdown():
        pub.publish(torque_command)
        rospy.loginfo("Publishing joint torque command: %s" % torque_command.effort)
        rate.sleep()

if __name__ == '__main__':
    try:
        joint_torque_publisher()
    except rospy.ROSInterruptException:
        pass
