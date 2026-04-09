# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for the extension."""

##
# Register Gym environments.
##


import gymnasium as gym

from .direct.byte_01_v1.byte_01_v1_env import Byte01V1Env
from .direct.byte_01_v1.byte_01_v1_env_cfg import Byte01V1EnvCfg

gym.register(
    id="Template-Byte-01-V1-Direct-v0",
    entry_point="byte_01_v1.tasks.direct.byte_01_v1.byte_01_v1_env:Byte01V1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "byte_01_v1.tasks.direct.byte_01_v1.byte_01_v1_env_cfg:Byte01V1EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "byte_01_v1.tasks.direct.byte_01_v1.agents.rsl_rl_ppo_cfg:PPORunnerCfg"
        ),
    },
)