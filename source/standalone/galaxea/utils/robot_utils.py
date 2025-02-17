import argparse
import re
import numpy as np
import pygame
import time
import rospy
import tf2_ros
import tf
from geometry_msgs.msg import TransformStamped
import geometry_msgs.msg
from tf2_ros import Buffer, TransformListener




import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
import threading
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from scipy.spatial.transform import Rotation as R


def pose_msg(trans, rot):
    target_msg = PoseStamped()
    target_msg.header.stamp = rospy.Time.now()
    target_msg.header.frame_id = "base_link"  # Adjust the frame ID as needed
    target_msg.pose.position.x = trans[0]
    target_msg.pose.position.y = trans[1]
    target_msg.pose.position.z = trans[2]
    target_msg.pose.orientation.x = rot[0]
    target_msg.pose.orientation.y = rot[1]
    target_msg.pose.orientation.z = rot[2]
    target_msg.pose.orientation.w = rot[3]
    return target_msg

def pose_to_matrix(pos, quat):
    """
    Convert position and quaternion to a homogeneous transformation matrix.
    
    Args:
        pos (list or np.array): [x, y, z] position.
        quat (list or np.array): [qx, qy, qz, qw] quaternion.
    
    Returns:
        np.array: 4x4 transformation matrix.
    """
    # Create a 3x3 rotation matrix from quaternion
    rot_matrix = R.from_quat(quat).as_matrix()
    
    # Create a 4x4 homogeneous transformation matrix
    matrix = np.eye(4)
    matrix[:3, :3] = rot_matrix
    matrix[:3, 3] = pos
    
    return matrix

def matrix_to_pose(matrix):
    """
    Convert a homogeneous transformation matrix to position and quaternion.
    
    Args:
        matrix (np.array): 4x4 transformation matrix.
    
    Returns:
        tuple: (position, quaternion)
    """
    pos = matrix[:3, 3]

    rot = R.from_matrix(matrix[:3, :3])
    quat = rot.as_quat()  # [qx, qy, qz, qw]
    return pos, quat


def generate_positions_commands(pick_positions, gripper_thickness, drop_positions, gripper_error):
    positions_commands = [
        ([0.3864, -0.5237, 1.1475], [1.0, 0.0, 0.0, 0.0], "o"),  # 开启夹爪
        ([0.3864, -0.4946, 1.1475], [1.0, 0.0, 0.0, 0.0], None),
        ([0.3864, -0.4452, 1.1475], [1.0, 0.0, 0.0, 0.0], None),
        ([0.3864, -0.4146, 1.1475], [1.0, 0.0, 0.0, 0.0], None),
        ([pick_positions[0,0], pick_positions[0,1], 1.1475], [1.0, 0.0, 0.0, 0.0], None),
        ([pick_positions[0,0], pick_positions[0,1], 1.1475], [0.9970, 0.0, 0.0768, 0.0], None),
        ([pick_positions[0,0], pick_positions[0,1], 1.1475], [0.9608, 0.0, 0.2773, 0.0], None),
        ([pick_positions[0,0], pick_positions[0,1], 1.1475], [0.8996, 0.0, 0.4367, 0.0], None),
        ([pick_positions[0,0], pick_positions[0,1], 1.1475], [0.7918, 0.0, 0.6108, 0.0], None),
        ([pick_positions[0,0], pick_positions[0,1], 1.1475], [0.7576, 0.0, 0.6527, 0.0], None),
        ([pick_positions[0,0], pick_positions[0,1], 1.1475], [0.7576, 0.0, 0.6527, 0.0], None),
        ([pick_positions[0,0], pick_positions[0,1], 1.0419], [0.7576, 0.0, 0.6527, 0.0], None),
        ([pick_positions[0,0], pick_positions[0,1], 1.0229], [0.7576, 0.0, 0.6527, 0.0], None),
        ([pick_positions[0,0], pick_positions[0,1], 1.01], [0.7576, 0.0, 0.6527, 0.0], None),
        (pick_positions + gripper_thickness, [0.7576, 0.0, 0.6527, 0.0], "c"),  # 夹爪闭合
        ([0.3756, -0.3657, 1.1], [0.7576, 0.0, 0.6527, 0.0], None),
        ([0.3756, -0.3657, 1.1], [0.7576, 0.0, 0.6527, 0.0], None),
        ([0.3756, -0.3657, 1.1801], [0.7576, 0.0, 0.6527, 0.0], None),
        ([0.3756, -0.3657, 1.2197], [0.7576, 0.0, 0.6527, 0.0], None),
        ([0.3756, -0.3657, 1.2699], [0.7576, 0.0, 0.6527, 0.0], None),
        ([0.3756, -0.3657, 1.2943], [0.7576, 0.0, 0.6527, 0.0], None),
        ([0.4029, -0.3657, 1.2943], [0.7576, 0.0, 0.6527, 0.0], None),
        ([0.4029, -0.2935, 1.2943], [0.7576, 0.0, 0.6527, 0.0], None),
        ([0.4029, -0.2389, 1.2943], [0.7576, 0.0, 0.6527, 0.0], None),
        ([0.4029, -0.2389, 1.3544], [0.7576, 0.0, 0.6527, 0.0], None),
        ([0.4029, -0.1695, 1.3544], [0.7576, 0.0, 0.6527, 0.0], None),
        ([0.4029, -0.1231, 1.3544], [0.7576, 0.0, 0.6527, 0.0], None),
        ([0.4029, -0.1131, 1.3544], [0.7576, 0.0, 0.6527, 0.0], None),
        (drop_positions + gripper_error , [0.7576, 0.0, 0.6527, 0.0], "o")   # 夹爪开启
    ]
    
    return positions_commands