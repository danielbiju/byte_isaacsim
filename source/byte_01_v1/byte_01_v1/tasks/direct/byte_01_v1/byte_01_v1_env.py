# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.math import sample_uniform, quat_from_euler_xyz, euler_xyz_from_quat
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG

from .byte_01_v1_env_cfg import Byte01V1EnvCfg


class Byte01V1Env(DirectRLEnv):
    """Direct RL locomotion environment for the Byte 01 V1 quadruped.

    BYTE-01 Leg Layout
    ------------------
    Front | FL = Leg 1 (hip1, thigh1, knee1, feet1)
          | FR = Leg 3 (hip3, thigh3, knee3, feet3)
    Rear  | RL = Leg 2 (hip2, thigh2, knee2, feet2)
          | RR = Leg 4 (hip4, thigh4, knee4, feet4)

    feet_ids index mapping -> feet1(FL)=0 feet2(RL)=1 feet3(FR)=2 feet4(RR)=3

    Trot diagonal pairs
    -------------------
    Diagonal A : FL(feet1=0) + RR(feet4=3) swing together
    Diagonal B : FR(feet3=2) + RL(feet2=1) swing together

    Observation space (48-dim):
    [base_lin_vel(3), base_ang_vel(3), projected_gravity(3),
     vel_commands(3), joint_pos_rel(12), joint_vel(12), last_actions(12)]

    Action space (12-dim):
    Joint position deltas [rad] -> target = default_pos + action_scale * action

    Termination:
    * base_link OR any hip (hip1-4) contact force > 1 N over history window
    * Base height drops below cfg.min_base_height
    * Episode timeout

    Reward shaping (all scale names match Byte01V1EnvCfg exactly):
    lin_vel_reward_scale          (+) axis-aware velocity tracking:
                                      x-cmd only  -> reward vel_x match only
                                      y-cmd only  -> reward vel_y match only
                                      both/neither -> combined x+y tracking
    yaw_rate_reward_scale         (+) yaw-rate tracking
    flat_orientation_reward_scale (+) upright base (Gaussian around 0 tilt)
    alive_reward_scale            (+) per-step survival bonus
    feet_air_time_reward_scale    (+) periodic foot stepping / gait quality
    trot_reward_scale             (+) diagonal synchronisation
    z_vel_reward_scale            (-) vertical base bouncing
    ang_vel_reward_scale          (-) roll / pitch rate
    joint_torque_reward_scale     (-) energy usage
    joint_accel_reward_scale      (-) joint jerk
    action_rate_reward_scale      (-) sudden action changes
    termination_reward_scale      (-) explicit death penalty
    excessive_air_time_scale      (-) foot held airborne too long
    foot_spread_scale             (-) foot tucked under body
    off_axis_lin_vel_penalty_scale(-) motion on non-commanded linear axes:
                                      x-cmd only  -> penalise vel_y + vel_z
                                      y-cmd only  -> penalise vel_x + vel_z
    """

    cfg: Byte01V1EnvCfg

    def __init__(self, cfg: Byte01V1EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # -- joint indices
        self._joint_ids, _ = self.robot.find_joints("revolute.*")
        self._hip_joint_ids = torch.tensor([0, 3, 6, 9], device=self.device)
        self._num_joints = len(self._joint_ids)

        # -- contact sensor body indices
        # Die bodies: base_link + all 4 hips.
        # Episode terminates if ANY of these registers > 1 N contact.
        # Index order -> hip1=0 hip2=1 hip3=2 hip4=3 thigh1=4 ... base_link=12
        self._die_body_ids, _ = self._contact_sensor.find_bodies(
            ["hip1", "hip2", "hip3", "hip4",
             "thigh1", "thigh2", "thigh3", "thigh4",
             "knee1", "knee2", "knee3", "knee4", "base_link"]
        )

        # Feet contact indices (contact sensor)
        # Index -> leg: 0=FL(feet1) 1=RL(feet2) 2=FR(feet3) 3=RR(feet4)
        #self._feet_ids, _ = self._contact_sensor.find_bodies(["feet1", "feet2", "feet3", "feet4"])

        # Feet body indices (articulation data -- for position queries)
        # Same order: feet1(FL)=0 feet2(RL)=1 feet3(FR)=2 feet4(RR)=3
        self._feet_body_ids, _ = self.robot.find_bodies(["feet1", "feet2", "feet3", "feet4"])
        feet_names = ["feet1", "feet2", "feet3", "feet4"]
        self._feet_ids, feet_names_found = self._contact_sensor.find_bodies(feet_names)

        # Ensure order consistency
        name_to_index = {name: i for i, name in enumerate(feet_names_found)}
        self._feet_ids = torch.tensor(
            [self._feet_ids[name_to_index[name]] for name in feet_names],
            device=self.device,
            dtype=torch.long,
        )
        self._num_feet = len(self._feet_ids)

        # -- default joint positions
        self._default_joint_pos = self.robot.data.default_joint_pos[
            :, self._joint_ids
        ].clone()

        # -- per-env state buffers
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        self._last_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._last_joint_vel = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        # Airborne timer: increments every step, resets to 0 on touchdown
        # Shape (N, 4): columns -> FL(feet1) RL(feet2) FR(feet3) RR(feet4)
        self._feet_air_time = torch.zeros(self.num_envs, self._num_feet, device=self.device)

        # Velocity commands [x_vel m/s, y_vel m/s, yaw_rate rad/s]
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Gaussian kernel width for flat-orientation reward
        self._tilt_sigma: float = math.sin(math.radians(self.cfg.max_tilt_angle_deg)) ** 2

    # --------------------------------------------------------------------------
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.terrain = TerrainImporter(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # --------------------------------------------------------------------------
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            # Green arrow = commanded velocity
            cmd_cfg = GREEN_ARROW_X_MARKER_CFG.replace(
                prim_path="/Visuals/Command/cmd_vel"
            )
            cmd_cfg.markers["arrow"].scale = (1.0, 0.2, 0.2)
            self._cmd_vel_visualizer = VisualizationMarkers(cmd_cfg)

            # Blue arrow = actual velocity
            act_cfg = BLUE_ARROW_X_MARKER_CFG.replace(
                prim_path="/Visuals/Command/act_vel"
            )
            act_cfg.markers["arrow"].scale = (1.0, 0.2, 0.2)
            self._act_vel_visualizer = VisualizationMarkers(act_cfg)
        else:
            if hasattr(self, "_cmd_vel_visualizer"):
                self._cmd_vel_visualizer.set_visibility(False)
            if hasattr(self, "_act_vel_visualizer"):
                self._act_vel_visualizer.set_visibility(False)

    # --------------------------------------------------------------------------
    def _debug_vis_callback(self, event):
        # Robot base positions — float up 0.6 m above base
        base_pos = self.robot.data.root_pos_w.clone()  # (N, 3)
        base_pos[:, 2] += 0.6

        # Safe yaw extraction from root quaternion (avoids heading_w attr issue)
        _, _, heading_w = euler_xyz_from_quat(self.robot.data.root_quat_w)  # (N,)

        # ── Commanded velocity arrow ───────────────────────────────────────────
        cmd_x = self._commands[:, 0]
        cmd_y = self._commands[:, 1]
        cmd_speed = torch.norm(self._commands[:, :2], dim=1).clamp(min=1e-3)

        # yaw angle of command vector in world frame
        cmd_yaw = torch.atan2(cmd_y, cmd_x) + heading_w  # (N,)

        cmd_quats = quat_from_euler_xyz(
            torch.zeros_like(cmd_yaw),
            torch.zeros_like(cmd_yaw),
            cmd_yaw,
        )  # (N, 4)

        cmd_scales = torch.stack(
            [cmd_speed.clamp(0.1, 2.0),
             torch.full_like(cmd_speed, 0.1),
             torch.full_like(cmd_speed, 0.1)],
            dim=1,
        )  # (N, 3)

        self._cmd_vel_visualizer.visualize(
            translations=base_pos,
            orientations=cmd_quats,
            scales=cmd_scales,
        )

        # ── Actual velocity arrow ──────────────────────────────────────────────
        act_lin = self.robot.data.root_lin_vel_w  # (N, 3) world frame
        act_x = act_lin[:, 0]
        act_y = act_lin[:, 1]
        act_speed = torch.norm(act_lin[:, :2], dim=1).clamp(min=1e-3)

        act_yaw = torch.atan2(act_y, act_x)  # (N,)

        act_quats = quat_from_euler_xyz(
            torch.zeros_like(act_yaw),
            torch.zeros_like(act_yaw),
            act_yaw,
        )

        act_scales = torch.stack(
            [act_speed.clamp(0.1, 2.0),
             torch.full_like(act_speed, 0.1),
             torch.full_like(act_speed, 0.1)],
            dim=1,
        )

        self._act_vel_visualizer.visualize(
            translations=base_pos,
            orientations=act_quats,
            scales=act_scales,
        )

    # --------------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        targets = self._default_joint_pos + self.cfg.action_scale * self.actions
        self.robot.set_joint_position_target(targets, joint_ids=self._joint_ids)

    # --------------------------------------------------------------------------
    def _get_observations(self) -> dict:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        joint_pos_rel = self.joint_pos[:, self._joint_ids] - self._default_joint_pos

        obs = torch.cat(
            [
                self.robot.data.root_lin_vel_b,          # (N, 3)
                self.robot.data.root_ang_vel_b,          # (N, 3)
                self.robot.data.projected_gravity_b,     # (N, 3)
                self._commands,                          # (N, 3)
                joint_pos_rel,                           # (N, 12)
                self.joint_vel[:, self._joint_ids],      # (N, 12)
                self._last_actions,                      # (N, 12)
            ],
            dim=-1,
        )  # -> (N, 48)

        return {"policy": obs}

    # --------------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        joint_vel_cur = self.joint_vel[:, self._joint_ids]
        joint_accel = (joint_vel_cur - self._last_joint_vel) / (
            self.cfg.sim.dt * self.cfg.decimation
        )
        joint_pos_rel = self.joint_pos[:, self._joint_ids] - self._default_joint_pos
        base_height = self.robot.data.root_pos_w[:, 2]

        # -- feet contact
        # Vertical (Z) net contact force, latest history frame.
        # Shape (N, 4) -> col 0=FL(feet1) col 1=RL(feet2) col 2=FR(feet3) col 3=RR(feet4)
        foot_forces = self._contact_sensor.data.net_forces_w_history[
            :, 0, self._feet_ids, 2
        ]
        feet_contact = foot_forces > 1.0  # (N, 4) bool

        # XY spread: distance of each foot from base CoM in the horizontal plane
        feet_pos_w = self.robot.data.body_pos_w[:, self._feet_body_ids, :]  # (N, 4, 3)
        base_pos_w = self.robot.data.root_pos_w                              # (N, 3)
        feet_rel_xy = feet_pos_w[:, :, :2] - base_pos_w[:, None, :2]        # (N, 4, 2)
        feet_xy_spread = torch.norm(feet_rel_xy, dim=-1)                     # (N, 4)

        total_reward = compute_rewards(
            self.robot.data.root_lin_vel_b,
            self.robot.data.root_ang_vel_b,
            self._commands,
            self.robot.data.projected_gravity_b,
            self._tilt_sigma,
            self.robot.data.applied_torque[:, self._joint_ids],
            joint_accel,
            self.actions,
            self._last_actions,
            feet_contact,
            self._feet_air_time,
            self.cfg.sim.dt * self.cfg.decimation,
            self.reset_terminated,
            self.cfg.lin_vel_reward_scale,
            self.cfg.yaw_rate_reward_scale,
            self.cfg.z_vel_reward_scale,
            self.cfg.ang_vel_reward_scale,
            self.cfg.joint_torque_reward_scale,
            self.cfg.joint_accel_reward_scale,
            self.cfg.action_rate_reward_scale,
            self.cfg.flat_orientation_reward_scale,
            self.cfg.alive_reward_scale,
            self.cfg.termination_reward_scale,
            self.cfg.feet_air_time_reward_scale,
            self.cfg.trot_reward_scale,
            feet_xy_spread,
            self.cfg.excessive_air_time_scale,
            self.cfg.foot_spread_scale,
            joint_pos_rel[:, self._hip_joint_ids],
            base_height,
            self.cfg.hip_default_scale,
            self.cfg.height_reward_scale,
            self.cfg.low_height_penalty_scale,
            self.cfg.yaw_spin_penalty_scale,
            self.cfg.short_air_time_scale,
            self.cfg.off_axis_lin_vel_penalty_scale,    # NEW
        )

        # -- roll forward buffers
        self._last_actions = self.actions.clone()
        self._last_joint_vel = joint_vel_cur.clone()

        # Air-time counter: add dt every step, zero-out on touchdown
        self._feet_air_time += self.cfg.sim.dt * self.cfg.decimation
        self._feet_air_time *= (~feet_contact).float()

        return total_reward

    # --------------------------------------------------------------------------
    def _get_dones(self):
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        net_contact_forces = self._contact_sensor.data.net_forces_w_history

        # Condition 1: any die body touches ground
        body_contact = torch.any(
            torch.max(
                torch.norm(net_contact_forces[:, :, self._die_body_ids], dim=-1),
                dim=1,
            )[0] > 1.0,
            dim=1,
        )

        # Condition 2: base collapsed (catches falls that don't involve hip contact)
        base_too_low = self.robot.data.root_pos_w[:, 2] < self.cfg.min_base_height

        died = (body_contact | base_too_low) & (self.episode_length_buf > 1)
        return died, time_out

    # --------------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        n = len(env_ids)

        self._commands[env_ids, 0] = sample_uniform(
            self.cfg.command_x_range[0], self.cfg.command_x_range[1], (n,), self.device
        )
        self._commands[env_ids, 1] = sample_uniform(
            self.cfg.command_y_range[0], self.cfg.command_y_range[1], (n,), self.device
        )
        self._commands[env_ids, 2] = sample_uniform(
            self.cfg.command_yaw_range[0], self.cfg.command_yaw_range[1], (n,), self.device
        )

        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._last_actions[env_ids] = 0.0
        self._last_joint_vel[env_ids] = 0.0
        self._feet_air_time[env_ids] = 0.0


# =============================================================================
# JIT-compiled reward kernel -- runs entirely on GPU
# =============================================================================
# BYTE-01 feet_contact column mapping
# col 0 -> feet1 = FL (Front-Left,  Leg 1)
# col 1 -> feet2 = RL (Rear-Left,   Leg 2)
# col 2 -> feet3 = FR (Front-Right, Leg 3)
# col 3 -> feet4 = RR (Rear-Right,  Leg 4)
#
# Trot diagonal pairs
# Diagonal A : FL(col 0) + RR(col 3) swing together
# Diagonal B : FR(col 2) + RL(col 1) swing together
# =============================================================================
@torch.jit.script
def compute_rewards(
    root_lin_vel_b: torch.Tensor,
    root_ang_vel_b: torch.Tensor,
    commands: torch.Tensor,
    projected_grav_b: torch.Tensor,
    tilt_sigma: float,
    torques: torch.Tensor,
    joint_accel: torch.Tensor,
    actions: torch.Tensor,
    last_actions: torch.Tensor,
    feet_contact: torch.Tensor,            # (N, 4) bool
    feet_air_time: torch.Tensor,           # (N, 4) seconds airborne
    dt: float,
    reset_terminated: torch.Tensor,
    lin_vel_reward_scale: float,
    yaw_rate_reward_scale: float,
    z_vel_reward_scale: float,
    ang_vel_reward_scale: float,
    joint_torque_reward_scale: float,
    joint_accel_reward_scale: float,
    action_rate_reward_scale: float,
    flat_orientation_reward_scale: float,
    alive_reward_scale: float,
    termination_reward_scale: float,
    feet_air_time_reward_scale: float,
    trot_reward_scale: float,
    feet_xy_spread: torch.Tensor,          # (N, 4)
    excessive_air_time_scale: float,
    foot_spread_scale: float,
    joint_pos_rel: torch.Tensor,           # (N, 12)
    base_height: torch.Tensor,             # (N,)
    hip_default_scale: float,
    height_reward_scale: float,
    low_height_penalty_scale: float,
    yaw_spin_penalty_scale: float,
    short_air_time_scale: float,
    off_axis_lin_vel_penalty_scale: float, # NEW — penalise non-commanded linear axes
) -> torch.Tensor:

    # ── command-mode detection ────────────────────────────────────────────────
    # Determine per-environment whether the command is purely on one axis.
    # Threshold 0.1 m/s filters out near-zero values from the sample_uniform range.
    cmd_x_active: torch.Tensor = torch.abs(commands[:, 0]) > 0.1   # (N,) bool
    cmd_y_active: torch.Tensor = torch.abs(commands[:, 1]) > 0.1   # (N,) bool

    x_only: torch.Tensor = cmd_x_active & (~cmd_y_active)   # pure x command
    y_only: torch.Tensor = cmd_y_active & (~cmd_x_active)   # pure y command

    # ── axis-aware velocity tracking reward ───────────────────────────────────
    # Per-axis squared errors
    x_err: torch.Tensor = torch.square(commands[:, 0] - root_lin_vel_b[:, 0])  # (N,)
    y_err: torch.Tensor = torch.square(commands[:, 1] - root_lin_vel_b[:, 1])  # (N,)
    combined_err: torch.Tensor = x_err + y_err                                  # (N,)

    # Select which error feeds the tracking reward:
    #   x-only  -> reward only x-axis match
    #   y-only  -> reward only y-axis match
    #   both/neither -> original combined x+y reward (backward-compatible)
    lin_vel_err: torch.Tensor = torch.where(
        x_only,
        x_err,
        torch.where(y_only, y_err, combined_err),
    )
    rew_lin_vel = torch.exp(-lin_vel_err / 0.25)

    # ── off-axis linear velocity penalty ─────────────────────────────────────
    # x-cmd only: penalise actual lateral (y) AND vertical (z) drift
    # y-cmd only: penalise actual forward (x) AND vertical (z) drift
    # This is ADDITIONAL to the existing global z_vel penalty (rew_z_vel),
    # providing a stronger, command-conditional suppression of unwanted motion.
    z_sq: torch.Tensor = torch.square(root_lin_vel_b[:, 2])
    rew_off_axis_lin_vel: torch.Tensor = (
        x_only.float() * (torch.square(root_lin_vel_b[:, 1]) + z_sq)
        + y_only.float() * (torch.square(root_lin_vel_b[:, 0]) + z_sq)
    )

    # ── yaw-rate tracking ─────────────────────────────────────────────────────
    yaw_err = torch.square(commands[:, 2] - root_ang_vel_b[:, 2])
    rew_yaw = torch.exp(-yaw_err / 0.25)

    # ── orientation ───────────────────────────────────────────────────────────
    orient_err = torch.sum(torch.square(projected_grav_b[:, :2]), dim=1)
    rew_flat_orient = torch.exp(-orient_err / tilt_sigma)

    # ── survival ──────────────────────────────────────────────────────────────
    rew_alive = 1.0 - reset_terminated.float()

    # ── air-time: Gaussian reward centred at 0.2 s ────────────────────────────
    AIR_TARGET: float = 0.4
    AIR_SIGMA: float = 0.08
    first_contact = (feet_air_time > 0.0) & feet_contact
    gauss_air = torch.exp(-torch.square(feet_air_time - AIR_TARGET) / AIR_SIGMA)
    cmd_norm = torch.norm(commands[:, :2], p=2, dim=1)
    moving = (cmd_norm > 0.1) | (root_lin_vel_b[:, 0] > 0.05)
    rew_air_time = torch.sum(gauss_air * first_contact.float(), dim=1) * moving.float()

    # ── short air-time penalty: punish < 0.1 s at touchdown ──────────────────
    too_short = torch.clamp(0.1 - feet_air_time, min=0.0) * first_contact.float()
    rew_short_air = torch.sum(too_short, dim=1) * moving.float()

    # ── excessive air-time penalty: punish > 0.4 s (ongoing) ─────────────────
    MAX_AIR: float = 0.8
    excessive_air = torch.clamp(feet_air_time - MAX_AIR, min=0.0)
    rew_excess_air = torch.sum(excessive_air, dim=1)

    # ── trot diagonal synchronisation ─────────────────────────────────────────
    fl_rr_contact = feet_contact[:, 0].float() + feet_contact[:, 3].float()  # FL + RR
    fr_rl_contact = feet_contact[:, 2].float() + feet_contact[:, 1].float()  # FR + RL
    trot_sync = torch.exp(-torch.square(torch.abs(fl_rr_contact - fr_rl_contact) - 2.0))
    rew_trot = trot_sync * (cmd_norm > 0.1).float()

    # ── height reward band ────────────────────────────────────────────────────
    rew_height = torch.clamp(base_height - 0.35, min=0.0)      # reward > 0.35 m
    rew_low_height = torch.clamp(0.30 - base_height, min=0.0)  # penalty < 0.30 m

    # ── leg abduction / hip default penalty ──────────────────────────────────
    rew_hip_default = torch.sum(torch.square(joint_pos_rel), dim=1)

    # ── direct yaw spin penalty (when no yaw command) ─────────────────────────
    no_yaw_cmd = (torch.abs(commands[:, 2]) < 0.1).float()
    rew_yaw_spin = torch.square(root_ang_vel_b[:, 2]) * no_yaw_cmd

    # ── standard penalties ────────────────────────────────────────────────────
    rew_termination = reset_terminated.float()
    rew_z_vel = torch.square(root_lin_vel_b[:, 2])
    rew_ang_vel_xy = torch.sum(torch.square(root_ang_vel_b[:, :2]), dim=1)
    rew_torque = torch.sum(torch.square(torques), dim=1)
    rew_joint_accel = torch.sum(torch.square(joint_accel), dim=1)
    rew_action_rate = torch.sum(torch.square(actions - last_actions), dim=1)

    MIN_SPREAD: float = 0.08
    tuck_violation = torch.clamp(MIN_SPREAD - feet_xy_spread, min=0.0)
    rew_foot_spread = torch.sum(tuck_violation, dim=1)

    # ── total ─────────────────────────────────────────────────────────────────
    total = (
        lin_vel_reward_scale             * rew_lin_vel
        + yaw_rate_reward_scale          * rew_yaw
        + flat_orientation_reward_scale  * rew_flat_orient
        + alive_reward_scale             * rew_alive
        + feet_air_time_reward_scale     * rew_air_time
        + trot_reward_scale              * rew_trot
        + height_reward_scale            * rew_height
        + termination_reward_scale       * rew_termination
        + z_vel_reward_scale             * rew_z_vel
        + ang_vel_reward_scale           * rew_ang_vel_xy
        + joint_torque_reward_scale      * rew_torque
        + joint_accel_reward_scale       * rew_joint_accel
        + action_rate_reward_scale       * rew_action_rate
        + excessive_air_time_scale       * rew_excess_air
        + foot_spread_scale              * rew_foot_spread
        + hip_default_scale              * rew_hip_default
        + low_height_penalty_scale       * rew_low_height
        + yaw_spin_penalty_scale         * rew_yaw_spin
        + short_air_time_scale           * rew_short_air
        + off_axis_lin_vel_penalty_scale * rew_off_axis_lin_vel  # NEW
    )
    return total