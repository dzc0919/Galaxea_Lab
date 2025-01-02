# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the differential inverse kinematics controller with the simulator.

The differential IK controller can be configured in different modes. It uses the Jacobians computed by
PhysX. This helps perform parallelized computation of the inverse kinematics.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/galaxea/basic/run_diff_ik.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import time

import numpy as np

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Tutorial on using the differential IK controller."
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
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab.controllers import (
    DifferentialIKController,
    DifferentialIKControllerCfg,
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
    GALAXEA_R1_FIXBASE_HIGH_PD_CFG,
    GALAXEA_R1_HIGH_PD_GRIPPER_CFG,
)  # isort:skip


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
    # articulation
    if args_cli.robot == "R1":
        robot = GALAXEA_R1_FIXBASE_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(
            f"Robot {args_cli.robot} is not supported. Valid: R1, R1StrongGripper"
        )
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """运行仿真循环，控制机器人按正弦波运动。"""
    robot = scene["robot"]
    left_arm_entity_cfg = SceneEntityCfg(
        "robot", joint_names=["left_arm_joint.*"], body_names=["left_arm_link6"]
    )
    right_arm_entity_cfg = SceneEntityCfg(
        "robot", joint_names=["right_arm_joint.*"], body_names=["right_arm_link6"]
    )
    torso_entity_cfg = SceneEntityCfg(
        "robot", joint_names=["torso_joint.*"], body_names=["torso_link4"]
    )
    # Resolving the scene entities
    left_arm_entity_cfg.resolve(scene)
    right_arm_entity_cfg.resolve(scene)
    torso_entity_cfg.resolve(scene)
    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.
    print("robot.is_fixed_base: ", robot.is_fixed_base)
    if robot.is_fixed_base:
        left_ee_jacobi_idx = left_arm_entity_cfg.body_ids[0] - 1
        right_ee_jacobi_idx = right_arm_entity_cfg.body_ids[0] - 1
    else:
        left_ee_jacobi_idx = left_arm_entity_cfg.body_ids[0]
        right_ee_jacobi_idx = right_arm_entity_cfg.body_ids[0]

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

    print("-------------------------------------------------")
    print("left body_ids: ", left_arm_entity_cfg.body_ids)
    print("left joint_ids: ", left_arm_joint_ids)
    print("left_ee_jacobi_idx: ", left_ee_jacobi_idx)
    print("right body_ids: ", right_arm_entity_cfg.body_ids)
    print("right joint_ids: ", right_arm_joint_ids)
    print("right_ee_jacobi_idx: ", right_ee_jacobi_idx)
    print("num_arm_joints: ", num_arm_joints)
    print("*************************************************")
    print("left gripper joint_ids: ", left_gripper_joint_ids)
    print("right gripper joint_ids: ", right_gripper_joint_ids)
    print("num gripper joints: ", num_gripper_joints)
    print("-------------------------------------------------")

    sim_dt = sim.get_physics_dt()
    count = 0
    max_joint_value = np.array([0.30, 0.40, 0, 0.30, 0.30, 1.65]) / 180.0 * np.pi *100
    min_joint_value = np.array([-0.30, 0.0, -0.9, -0.30, -0.30, 1.65]) / 180.0 * np.pi *100
    max_torso_joint_value = np.array([0.71, -1.432, -0.71, 0]) / 180.0 * np.pi *100
    min_torso_joint_value = np.array([0, 0, 0, 0]) / 180.0 * np.pi *100

    while simulation_app.is_running():
        # 计算正弦波目标位置
        time = count * sim_dt *10
        phase = time / 10.0 * 2 * np.pi
        joint_position = min_joint_value + 0.8 * (max_joint_value - min_joint_value) * (1 + np.sin(phase)) / 2
        torso_joint_position =  0.8 * (max_torso_joint_value) * abs((np.sin(phase))) / 2 - 0.8 * (max_torso_joint_value) * (0 + np.sin(phase)) / 2
        time_tensor = torch.tensor(time, dtype=torch.float32,device="cuda:0")  # 将 time 转换为 Tensor 类型
        target_position_left = torch.tensor(joint_position,dtype=torch.float32, device="cuda:0")
        target_position_right = torch.tensor(joint_position,dtype=torch.float32, device="cuda:0")
        target_position_torso = torch.tensor(torso_joint_position,dtype=torch.float32, device="cuda:0")
        # 设置机器人关节目标位置

        robot.set_joint_position_target(target_position_left, joint_ids=left_arm_entity_cfg.joint_ids)
        robot.set_joint_position_target(target_position_right, joint_ids=right_arm_entity_cfg.joint_ids)
        robot.set_joint_position_target(target_position_torso, joint_ids=torso_entity_cfg.joint_ids)
        print(f"Target position torso: {target_position_torso}, shape: {target_position_torso.shape}")
        print(f"Joint ids torso: {torso_entity_cfg.joint_ids}, length: {len(torso_entity_cfg.joint_ids)}")
        print("target_position_left: ", target_position_left)
        print("target_position_right: ", target_position_right)
        print("target_position_torso: ", target_position_torso)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        
        count += 1
def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
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
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()