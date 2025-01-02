#!/usr/bin/env python3
# @copyright Galaxea

import time
import rospy
import numpy as np
from sensor_msgs.msg import JointState

# 初始化 ROS 节点
rospy.init_node('position_command_publisher')

# 创建发布器，只使用一个发布器来控制左右两臂
pub = rospy.Publisher('/isaac_sim/joint_command', JointState, queue_size=10)

# 创建 JointState 消息
joint_state_position = JointState()

# 设置左右臂的关节名称
joint_state_position.name = [
    'left_arm_joint1', 'left_arm_joint2', 'left_arm_joint3', 'left_arm_joint4', 'left_arm_joint5', 'left_arm_joint6',
    'right_arm_joint1', 'right_arm_joint2', 'right_arm_joint3', 'right_arm_joint4', 'right_arm_joint5', 'right_arm_joint6'
]

# 设定关节值的最大值和最小值（单位：弧度）
max_joint_value = np.array([30, 40.0, 0, 30.0, 30.0, 165.0]) / 180.0 * np.pi
min_joint_value = np.array([-30, 0.0, -90, -30.0, -30.0, 165.0]) / 180.0 * np.pi

# 初始化存储轨迹的列表
trajectory = []

# 控制发布的频率
rate = rospy.Rate(10)
start_time = time.time()
while not rospy.is_shutdown():
    current_time = time.time()
    phase = (current_time - start_time) / 10.0 * 2 * np.pi

    # 左右臂的关节位置计算
    joint_position_left = min_joint_value + 0.8 * (max_joint_value - min_joint_value) * (1 + np.sin(phase)) / 2
    joint_position_right = min_joint_value + 0.8 * (max_joint_value - min_joint_value) * (1 + np.sin(phase)) / 2

    # 将左右臂的关节位置组合到一起
    joint_state_position.position = np.concatenate([joint_position_left, joint_position_right]).tolist()

    # 发布关节位置
    pub.publish(joint_state_position)

    # 存储轨迹
    trajectory.append(joint_state_position.position)

    # 控制发布频率
    rate.sleep()

    # 停止条件
    if phase > 5 * np.pi:
        break

# 可选：保存轨迹数据
# np.savez('joint_trajectory.npz', name=joint_state_position.name, position=trajectory)
