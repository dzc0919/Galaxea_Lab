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
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
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
    robot = scene["robot"]

    torso_entity_cfg = SceneEntityCfg(
        "robot", joint_names=["torso_joint.*"], body_names=["torso_link4"]
    )
    # Resolving the scene entities
    torso_entity_cfg.resolve(scene)
    sim_dt = sim.get_physics_dt()
    count = 0
    max_torso_joint_value = np.array([0.7150001525878906, -1.4320001602172852,
                        -0.7159996032714844, 0.0])  *1.5
    min_torso_joint_value = np.array([0.0, 0.0, 0.0, 0.0])  # 设置初始值为0
    down_time = 15.0
    up_time = 15.0
    pause_time = 3.0
    time_factor = 5.0
    # print("仿真已启动，请输入 'down' 来下降机器人躯干。")
    # # 等待用户输入 'down' 后才继续执行下降
    # user_input = ""
    # while user_input != "down":
    #     user_input = input("输入 'down' 来降低躯干: ").strip().lower()
    #     if user_input == "down":
    #         print("收到 'down' 输入，开始下降躯干。")
    while simulation_app.is_running():
        time = count * sim_dt * time_factor
        if time <= down_time:  # 下降down_time
            torso_joint_position = min_torso_joint_value + (max_torso_joint_value - min_torso_joint_value) * (time / down_time)  # 线性插值下降
        elif time <= down_time + pause_time:  # 停留pause_time
            torso_joint_position = max_torso_joint_value
        elif time <= down_time + pause_time + up_time:  # 上升up_time
            torso_joint_position = max_torso_joint_value - (max_torso_joint_value - min_torso_joint_value) * ((time - down_time - pause_time) / up_time)  # 线性插值上升
        else:
            break  # 停止运行
        target_position_torso = torch.tensor(torso_joint_position,dtype=torch.float32, device="cuda:0")
        torso_joint_position[3] = 0.1 * np.pi  # 设置第四个关节的目标位置
        # 设置机器人关节目标位置
        robot.set_joint_position_target(target_position_torso, joint_ids=torso_entity_cfg.joint_ids)
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