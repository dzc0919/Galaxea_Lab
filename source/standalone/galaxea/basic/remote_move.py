# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Launch Isaac Sim Simulator first."""

import argparse
import time

import numpy as np
import matplotlib.pyplot as plt

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Dummy move."
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
    GALAXEA_R1_BASE_CFG,
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
        robot = GALAXEA_R1_BASE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(
            f"Robot {args_cli.robot} is not supported. Valid: R1, R1StrongGripper"
        )
import pygame

import pygame

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    pygame.init()
    screen = pygame.display.set_mode((400, 300))  # 创建一个窗口捕获事件
    pygame.display.set_caption("Robot Control")

    robot = scene["robot"]
    base_entity_cfg = SceneEntityCfg(
        "robot", joint_names=["dummy_base.*"]
    )
    base_entity_cfg.resolve(scene)
    sim_dt = sim.get_physics_dt()

    base_joint_velocity = np.array([0.0, 0.0, 0.0])  # 初始化速度为零
    speed_increment = 0.05  # 增加加速度步长
    max_speed = 2.0  # 最大速度
    acceleration = 0.1  # 控制加速度

    print("[INFO]: 控制机器人速度: W/S控制前后, A/D控制左右, Q/E控制旋转")
    running = True
    while simulation_app.is_running() and running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        # 通过按住W加速
        if keys[pygame.K_w]:
            base_joint_velocity[0] = min(base_joint_velocity[0] + acceleration, max_speed)
        elif keys[pygame.K_s]:
            base_joint_velocity[0] = max(base_joint_velocity[0] - acceleration, -max_speed)

        # 控制左右
        if keys[pygame.K_a]:
            base_joint_velocity[1] = min(base_joint_velocity[1] + acceleration, max_speed)
        elif keys[pygame.K_d]:
            base_joint_velocity[1] = max(base_joint_velocity[1] - acceleration, -max_speed)

        # 控制旋转
        if keys[pygame.K_q]:
            base_joint_velocity[2] = min(base_joint_velocity[2] + acceleration, max_speed)
        elif keys[pygame.K_e]:
            base_joint_velocity[2] = max(base_joint_velocity[2] - acceleration, -max_speed)

        # 停止
        if keys[pygame.K_SPACE]:
            base_joint_velocity = np.array([0.0, 0.0, 0.0])

        # 设置机器人速度
        target_velocity_base = torch.tensor(
            base_joint_velocity, dtype=torch.float32, device="cuda:0"
        )
        robot.set_joint_velocity_target(target_velocity_base, joint_ids=base_entity_cfg.joint_ids)

        # 更新模拟器
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    pygame.quit()

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