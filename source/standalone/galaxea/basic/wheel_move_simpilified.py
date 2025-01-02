# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""


.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/galaxea/basic/run_diff_ik.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import time

import numpy as np
import matplotlib.pyplot as plt

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
    GALAXEA_R1_HIGH_PD_CFG,
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
        robot = GALAXEA_R1_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(
            f"Robot {args_cli.robot} is not supported. Valid: R1, R1StrongGripper"
        )
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot = scene["robot"]

    torso_entity_cfg = SceneEntityCfg(
        "robot", joint_names=["torso_joint.*"], body_names=["torso_link4"]
    )
    steer_motor_entity_cfg = SceneEntityCfg(
        "robot", joint_names=["steer_motor_joint.*"]
    )
    wheel_motor_entity_cfg = SceneEntityCfg(
        "robot", joint_names=["wheel_motor_joint.*"],body_names=["steer_motor_link3"]
    )
    # Resolving the scene entities

    torso_entity_cfg.resolve(scene)
    steer_motor_entity_cfg.resolve(scene)
    wheel_motor_entity_cfg.resolve(scene)
    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.
    print("robot.is_fixed_base: ", robot.is_fixed_base)
    # get left/right gripper joint ids

    sim_dt = sim.get_physics_dt()
    count = 0

    min_torso_joint_value = np.array([0, 0, 0, 0]) / 180.0 * np.pi *100
    steer_joint_value = np.array([0.9250663955407291, 0.7346690973447606, 0.6716847170203316]) /2
    zero_steer_joint_value = np.array([0.0, 0.0, 0.0])
    wheel_joint_value = np.array([0.2692006686470151, 0.32073197533142844, 0.2555014677061562])*130
    wheel_joint_velocity = wheel_joint_value

    target_velocity_wheel = torch.tensor(wheel_joint_velocity,dtype=torch.float32, device="cuda:0")
    robot.set_joint_velocity_target(target_velocity_wheel, joint_ids=wheel_motor_entity_cfg.joint_ids)




    while simulation_app.is_running():

        torso_joint_position = min_torso_joint_value
        steer_joint_position = steer_joint_value

        target_position_torso = torch.tensor(torso_joint_position,dtype=torch.float32, device="cuda:0")
        target_position_steer = torch.tensor(steer_joint_position,dtype=torch.float32, device="cuda:0")

        robot.set_joint_position_target(target_position_torso, joint_ids=torso_entity_cfg.joint_ids)
        robot.set_joint_position_target(target_position_steer, joint_ids=steer_motor_entity_cfg.joint_ids)

        print("target_velocity_wheel: ", target_velocity_wheel)
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