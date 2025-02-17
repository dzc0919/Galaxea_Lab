

"""Launch Isaac Sim Simulator first."""

import argparse
import re
import numpy as np
import pygame
from omni.isaac.lab.app import AppLauncher
import time
import rospy
import tf2_ros
import tf
from geometry_msgs.msg import TransformStamped
import geometry_msgs.msg
from tf2_ros import Buffer, TransformListener

# add argparse arguments
parser = argparse.ArgumentParser(
    description="relaxed IK controller."
)
parser.add_argument("--robot", type=str, default="R1", help="Name of the robot.")
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import (
    RigidObjectCfg,
    AssetBaseCfg,
)
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from omni.isaac.lab_assets import (
    GALAXEA_R1_IK_HIGH_PD_CFG,

)  # isort:skip

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
import threading
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from scipy.spatial.transform import Rotation as R

@configclass
class IkSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(color=(1.0, 1.0, 1.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75)),
    )

    # object
    target_frame_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TargetFrameLeft",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.3864, 0.5237, 1.1475),
            # rot=(9.6247e-05, 9.7698e-01, -2.1335e-01, 3.9177e-04),
            rot=(0.99, 0.0, 0.0, 0.0),
        ),
    )
    target_frame_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TargetFrameRight",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.3864, -0.5237, 1.1475),
            # rot=(9.6247e-05, 9.7698e-01, -2.1335e-01, 3.9177e-04),
            rot=(0.99, 0.0, 0.0, 0.0),
        ),
    )
    table = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.52, 0.0, 0.0),
            rot=(-0.70711, 0.0, 0.0, 0.70711),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"../../../extensions/omni.isaac.lab_assets/data/Props/table/desk.usd",
            scale=(0.09, 0.09, 0.09),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,  # 禁用重力，确保桌子不会因重力移动
            kinematic_enabled=True,
            retain_accelerations=False,  # 防止物体在碰撞后受到加速度影响
        ),
        ),
    )
    basket = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Basket",
        debug_vis=True,
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"../../../extensions/omni.isaac.lab_assets/data/Props/fruits/basket.usd",
            scale=(0.4, 0.4, 0.4),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.45, 0.05, 1.05),
            rot=(0.707, 0.0, 0.0, 0.707),
        ),
    )
    carrot = RigidObjectCfg(
        prim_path="/World/envs/env_.*/carrot",
        spawn=sim_utils.UsdFileCfg(
            # usd_path=f"/home/user/zhr_workspace/IsaacsimAsset/objects/fruits/fruit_v2/banana.usd",
            usd_path=f"../../../extensions/omni.isaac.lab_assets/data/Props/fruits/carrot.usd",
            scale=(0.3, 0.3, 0.3),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
                rigid_body_enabled=True
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.35, -0.35, 0.96),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    banana = RigidObjectCfg(
        prim_path="/World/envs/env_.*/banana",
        spawn=sim_utils.UsdFileCfg(
            # usd_path=f"/home/user/zhr_workspace/IsaacsimAsset/objects/fruits/fruit_v2/banana.usd",
            usd_path=f"../../../extensions/omni.isaac.lab_assets/data/Props/fruits/banana.usd",
            scale=(0.3, 0.3, 0.3),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
                rigid_body_enabled=True
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6, -0.6, 0.96),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    # articulation
    if args_cli.robot == "R1":
        robot = GALAXEA_R1_IK_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(
            f"Robot {args_cli.robot} is not supported. Valid: R1, R1StrongGripper"
        )

def init_ros_publisher():
    rospy.init_node('isaac_sim_target_publisher', anonymous=True)
    pub_l = rospy.Publisher('/motion_target/target_pose_arm_left', PoseStamped, queue_size=10)
    pub_r = rospy.Publisher('/motion_target/target_pose_arm_right', PoseStamped, queue_size=10)
    rospy.Subscriber("/relaxed_ik/joint_angle_solutions_right", JointState, joint_angle_right_callback)
    rospy.Subscriber("/relaxed_ik/joint_angle_solutions_left", JointState, joint_angle_left_callback)
    return pub_l,pub_r

extracted_msg_right_global = None
extracted_msg_left_global = None

def joint_angle_right_callback(msg):
    global extracted_msg_right_global
    extracted_msg_right_global = msg.position  # Assuming msg is of type JointState
    # rospy.loginfo("Received joint angles: %s", extracted_msg_global)
    # print("Received joint angles: ", extracted_msg_global)

def joint_angle_left_callback(msg):
    global extracted_msg_left_global
    extracted_msg_left_global = msg.position  # Assuming msg is of type JointState
    # rospy.loginfo("Received joint angles: %s", extracted_msg_global)
    # print("Received joint angles: ", extracted_msg_global)


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
    # pos[1] = -pos[1]  # 反转 y 分量
    # quat[0] = -quat[0]  # 反转四元数的 w 分量
    # pos = np.array(pos[0], -pos[1], pos[2])
    # quat = np.array(-quat[0], quat[1], quat[2], quat[3])
    return pos, quat
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, pub_l: rospy.Publisher, pub_r: rospy.Publisher , br, listener):

    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.

    # omni.isaac.lab.assets.articulation.articulation.Articulation
    robot = scene["robot"] 
    # 0 11 12
    base_link_id = robot.data.body_names.index("base_link")
    left_arm_base_link = robot.data.body_names.index("left_arm_base_link")
    right_arm_base_link = robot.data.body_names.index("right_arm_base_link")
    
    # omni.isaac.core.prims.xform_prim_view.XFormPrimView
    target_frame_left = scene["target_frame_left"]
    target_frame_right = scene["target_frame_right"]

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)


    # Define goals for the arm
    # init pose: 0.3864, 0.5237, 1.1475, 9.6247e-05,  9.7698e-01, -2.1335e-01,  3.9177e-04
    target_position_left, target_orientation_left = target_frame_left.get_local_poses()
    target_position_right, target_orientation_right = target_frame_right.get_local_poses()
    # base_pose =  robot.right_arm_base_link.get_local_poses()
    # base_link_pose = robot.get_local_pose()
    # print("base_pose: ", base_link_pose)
    # print("base_pose: ", base_pose)
    # Track the given command
    # Create buffers to store actions
    

    # Specify robot-specific parameters
    left_arm_entity_cfg = SceneEntityCfg(
        "robot", joint_names=["left_arm_joint.*"], body_names=["left_arm_link6"]
    )
    right_arm_entity_cfg = SceneEntityCfg(
        "robot", joint_names=["right_arm_joint.*"], body_names=["right_arm_link6"]
    )

    # Resolving the scene entities
    left_arm_entity_cfg.resolve(scene)
    right_arm_entity_cfg.resolve(scene)
    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.
    print("robot.is_fixed_base: ", robot.is_fixed_base)


    left_arm_joint_ids = left_arm_entity_cfg.joint_ids
    right_arm_joint_ids = right_arm_entity_cfg.joint_ids
    num_arm_joints = len(left_arm_joint_ids)
    if num_arm_joints != len(right_arm_joint_ids):
        raise ValueError("The number of left and right arm joints should be the same.")

    left_gripper_entity_cfg = SceneEntityCfg("robot", joint_names=["left_gripper_.*"])
    right_gripper_entity_cfg = SceneEntityCfg("robot", joint_names=["right_gripper_.*"])
    left_gripper_entity_cfg.resolve(scene)
    right_gripper_entity_cfg.resolve(scene)

    # get left/right gripper joint ids
    left_gripper_joint_ids = left_gripper_entity_cfg.joint_ids
    right_gripper_joint_ids = right_gripper_entity_cfg.joint_ids
    num_gripper_joints = len(left_gripper_joint_ids)
    if num_gripper_joints != len(right_gripper_joint_ids):
        raise ValueError(
            "The number of left and right gripper joints should be the same."
        )

    # Define torso entity configuration
    # torso_entity_cfg = SceneEntityCfg("robot", joint_names=["torso_joint.*"])
    # torso_entity_cfg.resolve(scene)

    print("-------------------------------------------------")
    print("left body_ids: ", left_arm_entity_cfg.body_ids)
    print("left joint_ids: ", left_arm_joint_ids)
    print("right body_ids: ", right_arm_entity_cfg.body_ids)
    print("right joint_ids: ", right_arm_joint_ids)
    print("num_arm_joints: ", num_arm_joints)
    print("*************************************************")
    print("left gripper joint_ids: ", left_gripper_joint_ids)
    print("right gripper joint_ids: ", right_gripper_joint_ids)
    print("num gripper joints: ", num_gripper_joints)
    print("-------------------------------------------------")

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    index = 0  # 当前轨迹点索引

    # 读取 npz 文件
    data = np.load("positions_commands.npz")
    target_pose   = data["positions"]       # 目标位置信息
    target_orien  = data["orientations"]    # 目标姿态信息
    actions       = np.array(data["actions"])  # 夹爪动作

    # 将 "o" 映射为 0.05（打开），"c" 映射为 0.0（关闭），其他设为 None
    gripper_state_history = np.array([
        0.05 if a == "o" else 0.0 if a == "c" else None
        for a in actions
    ])

    # 计算总步数（确保用的是目标位置信息）
    total_steps = len(target_pose)

    # 预先初始化当前目标数据（默认取第一个数据）
    current_position_right  = target_pose[0]
    current_orientation_right = target_orien[0]
    current_gripper_state   = gripper_state_history[0]
    while simulation_app.is_running():
        body_pos = robot.data.body_pos_w.squeeze(0)
        body_quat = robot.data.body_quat_w.squeeze(0)
        base_link_pos = body_pos[base_link_id].cpu().numpy()
        base_link_quat = body_quat[base_link_id].cpu().numpy()
        base_link_quat = np.roll(base_link_quat, shift=-1)
       
        # print("base_link_pos: ", base_link_pos)
        # print("base_link_pos: ", base_link_quat)
        left_arm_base_link_pos = body_pos[left_arm_base_link].cpu().numpy()
        left_arm_base_link_quat = body_quat[left_arm_base_link].cpu().numpy()
        left_arm_base_link_quat = np.roll(left_arm_base_link_quat, shift=-1) 

        right_arm_base_link_pos = body_pos[right_arm_base_link].cpu().numpy()
        right_arm_base_link_quat = body_quat[right_arm_base_link].cpu().numpy()
        right_arm_base_link_quat = np.roll(right_arm_base_link_quat, shift=-1)


        target_position_left, target_orientation_left = target_frame_left.get_world_poses()
        target_position_left = target_position_left.squeeze(0).cpu().numpy()
        target_orientation_left = target_orientation_left.squeeze(0).cpu().numpy()
        target_orientation_left = np.roll(target_orientation_left, shift=-1)
        # print("targrt_orientation_left: ", target_orientation_left)
        # target_position_right, target_orientation_right = target_frame_right.get_world_poses()

        if count % 15 == 0 and index < total_steps:
            current_position_right  = target_pose[index]
            current_orientation_right = target_orien[index]
            current_gripper_state   = gripper_state_history[index]
            
            # 根据当前夹爪状态发送指令
            if current_gripper_state is not None:
                gripper_tensor = torch.tensor([current_gripper_state], dtype=torch.float32, device="cuda:0")
                robot.set_joint_position_target(gripper_tensor, joint_ids=right_gripper_joint_ids)
                
                if current_gripper_state == 0.05:
                    print("🟢 Gripper opened")
                elif current_gripper_state == 0.0:
                    print("🔴 Gripper closed")
                    
            index += 1  # 切换到下一步轨迹数据
    
        print("current_position_right: ", current_position_right)
        print("current_orientation_right: ", current_orientation_right)
        target_position_right = torch.tensor(current_position_right, dtype=torch.float32, device="cuda:0").squeeze(0)
        target_orientation_right = torch.tensor(current_orientation_right, dtype=torch.float32, device="cuda:0").squeeze(0)

        
        target_position_right = target_position_right.squeeze(0).cpu().numpy()

        target_orientation_right = target_orientation_right.squeeze(0).cpu().numpy()
        target_orientation_right = np.roll(target_orientation_right, shift=-1)
        # print("targrt_orientation_right: ", target_orientation_right)
        left_arm_matrix = pose_to_matrix(left_arm_base_link_pos, left_arm_base_link_quat)
        right_arm_matrix = pose_to_matrix(right_arm_base_link_pos, right_arm_base_link_quat)
        target_left_matrix = pose_to_matrix(target_position_left, target_orientation_left)
        target_right_matrix = pose_to_matrix(target_position_right, target_orientation_right)

        # 计算逆变换
        left_arm_inv = np.linalg.inv(left_arm_matrix)
        right_arm_inv = np.linalg.inv(right_arm_matrix)

        # 计算相对变换
        left_arm_to_target = np.dot(left_arm_inv, target_left_matrix)
        right_arm_to_target = np.dot(right_arm_inv, target_right_matrix)

        left_trans, left_rot = matrix_to_pose(left_arm_to_target)
        # print("left_rot: ", left_rot)
        right_trans, right_rot = matrix_to_pose(right_arm_to_target)
        
        
        # Publish the message
        pub_r.publish(pose_msg(right_trans, right_rot))
        pub_l.publish(pose_msg(left_trans,left_rot))
        

        # 使用全局变量 extracted_msg_global
        global extracted_msg_left_global
        global extracted_msg_right_global
        if extracted_msg_right_global is not None:
        #     # 将 extracted_msg_global 转换为 torch.cuda.FloatTensor
            right_joint_pos_des = torch.tensor(extracted_msg_right_global,dtype=torch.float32, device="cuda:0")
            robot.set_joint_position_target(
                right_joint_pos_des, joint_ids=right_arm_joint_ids
            )
        if extracted_msg_left_global is not None:
        #     # 将 extracted_msg_global 转换为 torch.cuda.FloatTensor
            left_joint_pos_des = torch.tensor(extracted_msg_left_global,dtype=torch.float32, device="cuda:0")
            robot.set_joint_position_target(
                left_joint_pos_des, joint_ids=left_arm_joint_ids
            )
      
        # right_joint_ids = [5, 7, 9, 11, 13, 15]  # 右臂关节索引
        # robot_pos = robot.data.joint_pos  # 获取所有关节位置
        # right_arm_pos = robot_pos[0, right_joint_ids]  # 取出右臂的关节位置

        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)



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


def convert_to_right_hand_system(pose, quat):
    # 假设 pose 是 [x, y, z]，quat 是 [x, y, z, w]
    # 反转 x 分量来转换坐标系
    new_pose = np.array([pose[0], -pose[1], pose[2]])
    # 反转四元数的 x 分量来转换坐标系
    quat = np.array(quat)
    new_quat = np.array([-quat[0], quat[1], quat[2], quat[3]])
    return new_pose, new_quat

def convert_to_left_hand_system(pose, quat):
    # 反转 x 分量来转换坐标系
    new_pose = np.array([pose[0], -pose[1], pose[2]])
    # 反转四元数的 x 分量来转换坐标系
    quat = np.array(quat)
    new_quat = np.array([-quat[0], quat[1], quat[2], quat[3]])
    return new_pose, new_quat





def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)


    pub_l,pub_r = init_ros_publisher()

    br = tf2_ros.TransformBroadcaster()
    listener = tf.TransformListener()

    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = IkSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene, pub_l,pub_r, br, listener)



if __name__ == "__main__":
    # run the main function
    main()
   
    # close sim app
    simulation_app.close()