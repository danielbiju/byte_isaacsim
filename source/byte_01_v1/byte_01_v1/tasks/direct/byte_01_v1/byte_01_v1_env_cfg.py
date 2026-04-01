"""
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg

@configclass
class EventCfg:
    #Configuration for randomization.

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (1.0, 3.0),
            "operation": "add",
        },
    )

@configclass
class Byte01V1EnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 20.0
    # - spaces definition
    action_scale = 1
    action_space = 12
    observation_space = 48
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200, 
        render_interval=1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",  # Path to base_link in cloned envs
        history_length=5,
        update_period=0.005,  # Matches sim dt = 1/200
        track_air_time=True,  # Not needed for base
    )

    #events
    events: EventCfg = EventCfg()

    # robot(s)
    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=200, env_spacing=3.0, replicate_physics=True)

    # custom parameters/scales
    # - controllable joint
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"
    # - action scale
    #action_scale = 100.0  # [N]
    # - reward scales
    lin_vel_reward_scale = 5.0  # Increased for more movement reward
    yaw_rate_reward_scale = 0.5
    z_vel_reward_scale = -4.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -1e-5  # Reduced punishment
    joint_accel_reward_scale = -1e-7  # Reduced punishment
    action_rate_reward_scale = -0.01
    flat_orientation_reward_scale = 15.0
    max_tilt_angle_deg = 10.0
    # - reset states/conditions
    initial_pole_angle_range = [-0.25, 0.25]  # pole angle sample range on reset [rad]
    max_cart_pos = 3.0  # reset if cart exceeds this position [m]

"""

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.byte_01 import BYTE_01_CFG
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg

# =============================================================================
# TODO: Replace with your robot's asset config.
# Option A — built-in asset for quick testing:
#   from isaaclab_assets.robots.anymal import ANYMAL_C_CFG as BYTE_01_V1_CFG
# Option B — your own USD file:
#   BYTE_01_V1_CFG = ArticulationCfg(
#       spawn=sim_utils.UsdFileCfg(usd_path="<path_to_your_robot.usd>"),
#       init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.55)),
#       actuators={...},
#   )
# =============================================================================


@configclass
class EventCfg:
    """Domain randomization applied at startup (sim-to-real robustness)."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (1.0, 3.0),
            "operation": "add",
        },
    )


@configclass
class Byte01V1EnvCfg(DirectRLEnvCfg):

    # ── env ───────────────────────────────────────────────────────────────────
    decimation: int         = 4
    episode_length_s: float = 20.0
    action_scale: float     = 1.0
    action_space: int       = 12
    observation_space: int  = 48
    state_space: int        = 0

    # ── simulation ────────────────────────────────────────────────────────────
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=4,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # ── terrain ───────────────────────────────────────────────────────────────
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        env_spacing=3.0,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # ── contact sensor ────────────────────────────────────────────────────────
    # Covers ALL robot bodies so we can filter by name in env.py.
    # track_air_time=True is required for feet_air_time reward.
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/kutta/.*",
        history_length=5,
        update_period=0.005,
        track_air_time=True,
    )

    # ── domain randomization ──────────────────────────────────────────────────
    events: EventCfg = EventCfg()

    # ── robot ─────────────────────────────────────────────────────────────────
    robot_cfg: ArticulationCfg = BYTE_01_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # ── scene ─────────────────────────────────────────────────────────────────
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=200,
        env_spacing=3.0,
        replicate_physics=True,
    )

    # ── velocity command ranges ───────────────────────────────────────────────
    command_x_range:   list = [0.7, 1.5]
    command_y_range:   list = [0.0, 0.0]
    command_yaw_range: list = [0.0, 0.0]

    # ── reward scales ─────────────────────────────────────────────────────────
    # --- positive rewards ----------------------------------------------------
    lin_vel_reward_scale:          float =  1.0
    yaw_rate_reward_scale:         float =  0.5
    flat_orientation_reward_scale: float =  5.0
    alive_reward_scale:            float =  2.5
    feet_air_time_reward_scale:    float =  3.0
    # --- trot reward ------------------------------------------------------
    trot_reward_scale:             float = 0.5

    # --- negative penalties ---------------------------------------------------
    z_vel_reward_scale:            float = -2.0
    ang_vel_reward_scale:          float = -0.05
    joint_torque_reward_scale:     float = -1e-5
    joint_accel_reward_scale:      float = -1e-7
    action_rate_reward_scale:      float = -0.01
    termination_reward_scale:      float = -5.0

    # ── orientation threshold ─────────────────────────────────────────────────
    max_tilt_angle_deg: float = 60.0