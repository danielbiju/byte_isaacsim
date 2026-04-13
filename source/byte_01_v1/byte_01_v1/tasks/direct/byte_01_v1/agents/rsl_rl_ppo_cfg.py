# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 64
    max_iterations = 30000
    save_interval = 100
    experiment_name = "byte_01_v1_walk"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True, #False (we found that normalization is not necessary for this task, and can actually hurt performance, likely due to the presence of binary contact sensors in the observation space)
        critic_obs_normalization=True, #False (same as above)
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5, #1.0 (we found that reducing the value loss coefficient can improve performance, likely by allowing the policy to focus more on maximizing reward rather than fitting the value function perfectly)
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01, #0.005 (we found that a small amount of entropy regularization can help performance, likely by encouraging more exploration in the early stages of training)
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4, #1.0e-3 (we found that a smaller learning rate can improve performance, likely by allowing for more stable updates)
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )