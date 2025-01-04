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
import random

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot = scene["robot"]
    base_entity_cfg = SceneEntityCfg(
        "robot", joint_names=["dummy_base.*"]
    )
    base_entity_cfg.resolve(scene)
    sim_dt = sim.get_physics_dt()

    # 初始化运动路径记录
    path = []  # 存储机器人每一步的速度和旋转

    print("[INFO]: 随机走动，最后原路返回")

    move_steps = 8  # 随机移动的总步数（次数）
    step_duration = 40  # 每步的持续时间（模拟步数）

    # 随机移动阶段
    for _ in range(move_steps):
        # 随机生成线速度方向
        random_velocity = np.array([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)])
        linear_velocity = random_velocity[:2] / np.linalg.norm(random_velocity[:2]) * 1.5  # 控制线速度大小
        angular_velocity = random_velocity[2] * 3  # 控制旋转速度大小

        # 合成速度（前两个分量为线速度，第三个分量为旋转速度）
        velocity = np.array([linear_velocity[0], linear_velocity[1], angular_velocity])

        # 记录路径（旋转速度取反）
        path.append(-velocity)

        # 设置速度
        target_velocity_base = torch.tensor(velocity, dtype=torch.float32, device="cuda:0")
        robot.set_joint_velocity_target(target_velocity_base, joint_ids=base_entity_cfg.joint_ids)

        # 持续模拟
        for _ in range(step_duration):
            # 更新模拟器
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

    print("[INFO]: 随机移动结束，开始原路返回")

    # 原路返回阶段
    for velocity in reversed(path):  # 按记录的路径反向返回
        target_velocity_base = torch.tensor(velocity, dtype=torch.float32, device="cuda:0")
        robot.set_joint_velocity_target(target_velocity_base, joint_ids=base_entity_cfg.joint_ids)

        # 持续模拟
        for _ in range(step_duration):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

    print("[INFO]: 已返回原点，模拟结束")
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