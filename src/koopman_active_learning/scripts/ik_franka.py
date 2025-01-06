#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from PyKDL import Chain, ChainFkSolverPos_recursive, ChainIkSolverPos_LMA, Frame, Rotation, Vector, JntArray
from kdl_parser_py.urdf import treeFromParam

class IKSolverNode:
    def __init__(self):
        rospy.init_node('kdl_ik_solver_py')

        robot_description = rospy.get_param('/robot_description')
        (ok, self.tree) = treeFromParam(robot_description)
        if not ok:
            rospy.logerr("Failed to construct kdl tree")
            exit(-1)

        self.chain = self.tree.getChain("panda_link0", "panda_link7")
        self.fk_solver = ChainFkSolverPos_recursive(self.chain)
        self.ik_solver = ChainIkSolverPos_LMA(self.chain)

        self.pose_sub = rospy.Subscriber("desired_pose", Pose, self.pose_callback)
        self.joint_pub = rospy.Publisher("joint_positions", JointState, queue_size=10)

    def pose_callback(self, msg):
        desired_pose = Frame(Rotation.Quaternion(msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w),
                             Vector(msg.position.x, msg.position.y, msg.position.z))

        joint_positions = JntArray(self.chain.getNrOfJoints())
        ret = self.ik_solver.CartToJnt(joint_positions, desired_pose, joint_positions)
        
        if ret >= 0:
            joint_state = JointState()
            joint_state.header.stamp = rospy.Time.now()
            joint_state.name = [''] * self.chain.getNrOfJoints()  # Optionally fill with actual joint names
            joint_state.position = [joint_positions[i] for i in range(self.chain.getNrOfJoints())]

            self.joint_pub.publish(joint_state)
        else:
            rospy.logerr("IK solver failed")

if __name__ == "__main__":
    ik_solver_node = IKSolverNode()
    rospy.spin()
