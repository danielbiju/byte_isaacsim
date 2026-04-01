'''
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .byte_01_v1_env_cfg import Byte01V1EnvCfg


class Byte01V1Env(DirectRLEnv):
    cfg: Byte01V1EnvCfg

    def __init__(self, cfg: Byte01V1EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._cart_dof_idx, _ = self.robot.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.robot.find_joints(self.cfg.pole_dof_name)

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        self.robot.set_joint_effort_target(self.actions * self.cfg.action_scale, joint_ids=self._cart_dof_idx)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_pole_pos,
            self.cfg.rew_scale_cart_vel,
            self.cfg.rew_scale_pole_vel,
            self.joint_pos[:, self._pole_dof_idx[0]],
            self.joint_vel[:, self._pole_dof_idx[0]],
            self.joint_pos[:, self._cart_dof_idx[0]],
            self.joint_vel[:, self._cart_dof_idx[0]],
            self.reset_terminated,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
            joint_pos[:, self._pole_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    return total_reward

'''

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
from isaaclab.utils.math import sample_uniform

from .byte_01_v1_env_cfg import Byte01V1EnvCfg


class Byte01V1Env(DirectRLEnv):
    """Direct RL locomotion environment for the Byte 01 V1 quadruped.

    Observation space (48-dim):
        [base_lin_vel(3), base_ang_vel(3), projected_gravity(3),
         vel_commands(3), joint_pos_rel(12), joint_vel(12), last_actions(12)]

    Action space (12-dim):
        Joint position deltas [rad] -> target = default_pos + action_scale * action

    Termination:
        * Thigh contact: any of thigh1-4 contact force > 1 N over history window
        * Episode timeout

    Reward shaping (all scale names match Byte01V1EnvCfg exactly):
        lin_vel_reward_scale          (+) forward/lateral velocity tracking
        yaw_rate_reward_scale         (+) yaw-rate tracking
        flat_orientation_reward_scale (+) upright base (Gaussian around 0 tilt)
        alive_reward_scale            (+) per-step survival bonus
        feet_air_time_reward_scale    (+) periodic foot stepping / gait quality
        z_vel_reward_scale            (-) vertical base bouncing
        ang_vel_reward_scale          (-) roll / pitch rate
        joint_torque_reward_scale     (-) energy usage
        joint_accel_reward_scale      (-) joint jerk
        action_rate_reward_scale      (-) sudden action changes
        termination_reward_scale      (-) explicit death penalty
    """

    cfg: Byte01V1EnvCfg

    def __init__(self, cfg: Byte01V1EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ── joint indices ─────────────────────────────────────────────────────
        # TODO: Replace ".*" with your exact 12 joint names from your USD/URDF.
        # Order should be: FL, FR, RL, RR x (hip, thigh, knee)
        # e.g. ["FL_hip_joint", "FL_thigh_joint", "FL_knee_joint", ...]
        self._joint_ids, _ = self.robot.find_joints("revolute.*")
        self._num_joints   = len(self._joint_ids)

        # ── contact sensor body indices ───────────────────────────────────────
        self._base_id, _      = self._contact_sensor.find_bodies("base_link")
        self._die_body_ids, _ = self._contact_sensor.find_bodies(
            ["thigh1", "thigh2", "thigh3", "thigh4"]
        )
        # TODO: Update foot body names to match your USD/URDF foot link names.
        self._feet_ids, _     = self._contact_sensor.find_bodies(
            ["knee1", "knee2", "knee3", "knee4"]
        )
        self._num_feet = len(self._feet_ids)

        # ── default joint positions ───────────────────────────────────────────
        self._default_joint_pos = self.robot.data.default_joint_pos[
            :, self._joint_ids
        ].clone()

        # ── per-env state buffers ─────────────────────────────────────────────
        self.joint_pos       = self.robot.data.joint_pos
        self.joint_vel       = self.robot.data.joint_vel
        self._last_actions   = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._last_joint_vel = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        # Tracks how long each foot has been airborne (reset on touchdown)
        self._feet_air_time  = torch.zeros(self.num_envs, self._num_feet, device=self.device)

        # Velocity commands [x_vel m/s, y_vel m/s, yaw_rate rad/s]
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Gaussian kernel width for flat-orientation reward
        # At exactly max_tilt_angle_deg, reward = exp(-1) ~ 0.37
        self._tilt_sigma: float = math.sin(math.radians(self.cfg.max_tilt_angle_deg)) ** 2

    # ─────────────────────────────────────────────────────────────────────────
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        import omni.usd
        from pxr import PhysxSchema, UsdPhysics
        stage = omni.usd.get_context().get_stage()

        print("\n=== RIGID BODY PRIMS ===")
        for prim in stage.TraverseAll():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                if not prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                    api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
                    api.CreateThresholdAttr(0.0)
                print(f"  {prim.GetPath()}")  # ← shows real prim paths
        print("========================\n")

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.terrain         = TerrainImporter(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"]    = self.robot
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
    # ─────────────────────────────────────────────────────────────────────────
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        targets = self._default_joint_pos + self.cfg.action_scale * self.actions
        self.robot.set_joint_position_target(targets, joint_ids=self._joint_ids)

    # ─────────────────────────────────────────────────────────────────────────
    def _get_observations(self) -> dict:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel


        joint_pos_rel = self.joint_pos[:, self._joint_ids] - self._default_joint_pos

        obs = torch.cat(
            [
                self.robot.data.root_lin_vel_b,       # (N,  3) lin vel  in base frame
                self.robot.data.root_ang_vel_b,       # (N,  3) ang vel  in base frame
                self.robot.data.projected_gravity_b,  # (N,  3) gravity  in base frame
                self._commands,                       # (N,  3) [x_vel, y_vel, yaw_rate]
                joint_pos_rel,                        # (N, 12) relative joint positions
                self.joint_vel[:, self._joint_ids],   # (N, 12) joint velocities
                self._last_actions,                   # (N, 12) previous policy action
            ],
            dim=-1,
        )  # -> (N, 48)
        if self.episode_length_buf[10] == 11:
            print(f"Base height at step 11: {self.robot.data.root_pos_w[0, 2].item():.4f} m")

        return {"policy": obs}

    # ─────────────────────────────────────────────────────────────────────────
    def _get_rewards(self) -> torch.Tensor:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        joint_vel_cur = self.joint_vel[:, self._joint_ids]
        joint_accel   = (joint_vel_cur - self._last_joint_vel) / (
            self.cfg.sim.dt * self.cfg.decimation
        )

        # ── feet contact from sensor ──────────────────────────────────────────
        # net_forces_w_history shape: (N, history_length, num_sensor_bodies, 3)
        # Take the latest frame [index 0] for contact detection
        foot_forces  = self._contact_sensor.data.net_forces_w_history[
            :, 0, self._feet_ids, 2
        ]                                           # (N, num_feet) — vertical force
        feet_contact = foot_forces > 1.0            # (N, num_feet) bool

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
        )

        # ── roll forward buffers ──────────────────────────────────────────────
        self._last_actions   = self.actions.clone()
        self._last_joint_vel = joint_vel_cur.clone()

        # Air-time counter: increment every step, reset to 0 on touchdown
        self._feet_air_time += self.cfg.sim.dt * self.cfg.decimation
        self._feet_air_time *= (~feet_contact).float()

        return total_reward

    # ─────────────────────────────────────────────────────────────────────────
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # ── Termination 1: base height too low = robot collapsed ─────────────
        # root_pos_w[:, 2] is the world-frame Z height of the base_link
        base_height = self.robot.data.root_pos_w[:, 2]
        base_too_low = base_height < 0.15   # tune this to your robot's geometry

        # ── Termination 2: tilt beyond 60° ───────────────────────────────────
        tilt = torch.norm(self.robot.data.projected_gravity_b[:, :2], dim=-1)
        tilt_too_much = tilt > math.sin(math.radians(60.0))

        # Grace period to let spawn physics settle
        grace = self.episode_length_buf > 10

        died = (base_too_low | tilt_too_much) & grace

        return died, time_out

    # ─────────────────────────────────────────────────────────────────────────
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        n = len(env_ids)

        # Re-sample velocity commands
        self._commands[env_ids, 0] = sample_uniform(
            self.cfg.command_x_range[0],   self.cfg.command_x_range[1],   (n,), self.device
        )
        self._commands[env_ids, 1] = sample_uniform(
            self.cfg.command_y_range[0],   self.cfg.command_y_range[1],   (n,), self.device
        )
        self._commands[env_ids, 2] = sample_uniform(
            self.cfg.command_yaw_range[0], self.cfg.command_yaw_range[1], (n,), self.device
        )

        # Reset robot state to default pose at env origin
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Clear buffers for reset envs
        self._last_actions[env_ids]   = 0.0
        self._last_joint_vel[env_ids] = 0.0
        self._feet_air_time[env_ids]  = 0.0


# =============================================================================
# JIT-compiled reward kernel — runs entirely on GPU
# All parameter names mirror Byte01V1EnvCfg field names exactly.
# =============================================================================
@torch.jit.script
def compute_rewards(
    root_lin_vel_b:    torch.Tensor,   # (N, 3)
    root_ang_vel_b:    torch.Tensor,   # (N, 3)
    commands:          torch.Tensor,   # (N, 3) [cmd_x, cmd_y, cmd_yaw]
    projected_grav_b:  torch.Tensor,   # (N, 3)
    tilt_sigma:        float,
    torques:           torch.Tensor,   # (N, 12)
    joint_accel:       torch.Tensor,   # (N, 12)
    actions:           torch.Tensor,   # (N, 12)
    last_actions:      torch.Tensor,   # (N, 12)
    feet_contact:      torch.Tensor,   # (N, num_feet) bool
    feet_air_time:     torch.Tensor,   # (N, num_feet) seconds airborne
    dt:                float,
    reset_terminated:  torch.Tensor,   # (N,) bool
    # ── scales ────────────────────────────────────────────────────────────────
    lin_vel_reward_scale:          float,   #  5.0
    yaw_rate_reward_scale:         float,   #  0.5
    z_vel_reward_scale:            float,   # -4.0
    ang_vel_reward_scale:          float,   # -0.05
    joint_torque_reward_scale:     float,   # -1e-5
    joint_accel_reward_scale:      float,   # -1e-7
    action_rate_reward_scale:      float,   # -0.01
    flat_orientation_reward_scale: float,   #  15.0
    alive_reward_scale:            float,   #  0.5
    termination_reward_scale:      float,   # -5.0
    feet_air_time_reward_scale:    float,   #  1.0
    trot_reward_scale:             float,   #  1.0
) -> torch.Tensor:

    # ── (+) linear velocity tracking (exponential kernel) ────────────────────
    lin_vel_err = (
        torch.square(commands[:, 0] - root_lin_vel_b[:, 0])
        + torch.square(commands[:, 1] - root_lin_vel_b[:, 1])
    )
    rew_lin_vel = torch.exp(-lin_vel_err / 0.25)

    # ── (+) yaw rate tracking (exponential kernel) ────────────────────────────
    yaw_err  = torch.square(commands[:, 2] - root_ang_vel_b[:, 2])
    rew_yaw  = torch.exp(-yaw_err / 0.25)

    # ── (+) flat orientation ──────────────────────────────────────────────────
    # projected_grav_b = [0,0,-1] when upright; reward peaks when xy = 0
    # Gaussian: width set by tilt_sigma = sin^2(max_tilt_angle_deg)
    orient_err      = torch.sum(torch.square(projected_grav_b[:, :2]), dim=1)
    rew_flat_orient = torch.exp(-orient_err / tilt_sigma)

    # ── (+) alive bonus ───────────────────────────────────────────────────────
    rew_alive = 1.0 - reset_terminated.float()

    # ── (+) feet air time (gait quality) ─────────────────────────────────────
    # Fires once per foot at the instant of touchdown after a swing phase.
    # (air_time - 0.5): positive when foot was up > 0.5 s, else negative.
    # Gated by cmd_norm so it only fires when the robot is commanded to move.
    first_contact  = (feet_air_time > 0.0) & feet_contact
    air_reward     = torch.clamp(feet_air_time - 0.3, min=0.0) * first_contact.float()
    #               0.3s minimum air time before reward fires — filters tiny hops
    # Gate: reward stepping ONLY when commanded to move AND actually moving
    cmd_norm       = torch.norm(commands[:, :2], dim=1)
    moving         = (cmd_norm > 0.1) & (root_lin_vel_b[:, 0] > 0.05)
    #  ↑ robot must actually be moving forward
    rew_air_time   = torch.sum(air_reward, dim=1) * moving.float()
    # ── (+) encourage diagonal gait (trot pattern FL+RR, FR+RL) ──────────────
    # feet order assumed: [FL, FR, RL, RR] = indices [0, 1, 2, 3]
    fl_rr_contact = feet_contact[:, 0].float() + feet_contact[:, 3].float()
    fr_rl_contact = feet_contact[:, 1].float() + feet_contact[:, 2].float()
    # Reward when diagonal pairs are in sync (both near 1.0 or both near 0.0)
    trot_sync     = torch.exp(-torch.square(fl_rr_contact - fr_rl_contact))
    rew_trot      = trot_sync * (cmd_norm > 0.1).float()
  

    # ── (-) termination penalty ───────────────────────────────────────────────
    rew_termination = reset_terminated.float()

    # ── (-) vertical base velocity ────────────────────────────────────────────
    rew_z_vel       = torch.square(root_lin_vel_b[:, 2])

    # ── (-) roll / pitch angular velocity ────────────────────────────────────
    rew_ang_vel_xy  = torch.sum(torch.square(root_ang_vel_b[:, :2]), dim=1)

    # ── (-) joint torques ─────────────────────────────────────────────────────
    rew_torque      = torch.sum(torch.square(torques), dim=1)

    # ── (-) joint acceleration ────────────────────────────────────────────────
    rew_joint_accel = torch.sum(torch.square(joint_accel), dim=1)

    # ── (-) action rate ───────────────────────────────────────────────────────
    rew_action_rate = torch.sum(torch.square(actions - last_actions), dim=1)

    # ── combine all terms ─────────────────────────────────────────────────────
    total = (
          lin_vel_reward_scale          * rew_lin_vel
        + yaw_rate_reward_scale         * rew_yaw
        + flat_orientation_reward_scale * rew_flat_orient
        + alive_reward_scale            * rew_alive
        + feet_air_time_reward_scale    * rew_air_time
        + termination_reward_scale      * rew_termination
        + z_vel_reward_scale            * rew_z_vel
        + ang_vel_reward_scale          * rew_ang_vel_xy
        + joint_torque_reward_scale     * rew_torque
        + joint_accel_reward_scale      * rew_joint_accel
        + action_rate_reward_scale      * rew_action_rate
        + trot_reward_scale             * rew_trot
    )
    return total
