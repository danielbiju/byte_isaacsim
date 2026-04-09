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
# from isaaclab_assets.robots.anymal import ANYMAL_C_CFG as BYTE_01_V1_CFG
# Option B — your own USD file:
# BYTE_01_V1_CFG = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(usd_path=""),
#     init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.55)),
#     actuators={...},
# )
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
    decimation: int = 4
    episode_length_s: float = 20.0
    action_scale: float = 0.5
    action_space: int = 12
    observation_space: int = 48
    state_space: int = 0
    debug_vis: bool = True

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
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/kutta1/.*",
        history_length=5,
        update_period=0.005,
        track_air_time=True,
    )

    # ── domain randomization ──────────────────────────────────────────────────
    events: EventCfg = EventCfg()

    # ── robot ─────────────────────────────────────────────────────────────────
    robot_cfg: ArticulationCfg = BYTE_01_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=BYTE_01_CFG.spawn.replace(activate_contact_sensors=True),
    )

    # ── scene ─────────────────────────────────────────────────────────────────
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=200,
        env_spacing=3.0,
        replicate_physics=True,
    )

    # ── velocity command ranges ───────────────────────────────────────────────
    command_x_range: tuple = (0.7, 1.5)
    command_y_range: tuple = (0.0, 0.0)
    command_yaw_range: tuple = (0.0, 0.0)

    # ── reward scales — positive ──────────────────────────────────────────────
    lin_vel_reward_scale: float = 4.0
    yaw_rate_reward_scale: float = 2.0
    flat_orientation_reward_scale: float = 2.5
    alive_reward_scale: float = 1.0
    feet_air_time_reward_scale: float = 2.0
    trot_reward_scale: float = 5.0
    height_reward_scale: float = 2.0

    # ── reward scales — negative ──────────────────────────────────────────────
    z_vel_reward_scale: float = -2.0
    ang_vel_reward_scale: float = -0.05
    joint_torque_reward_scale: float = -1e-5
    joint_accel_reward_scale: float = -1e-7
    action_rate_reward_scale: float = -0.01
    termination_reward_scale: float = -5.0
    excessive_air_time_scale: float = -3.5
    short_air_time_scale: float = -2.5
    foot_spread_scale: float = -4.0
    hip_default_scale: float = -1.5
    low_height_penalty_scale: float = -3.0
    yaw_spin_penalty_scale: float = -4.0
    # Penalise motion on the two non-commanded linear axes when a pure
    # x-only or y-only velocity command is active.
    # x-cmd only  ->  penalise actual vel_y + vel_z
    # y-cmd only  ->  penalise actual vel_x + vel_z
    off_axis_lin_vel_penalty_scale: float = -2.0

    # ── thresholds ────────────────────────────────────────────────────────────
    max_tilt_angle_deg: float = 40.0
    min_base_height: float = 0.30