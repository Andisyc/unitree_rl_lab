from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float, command_name: str = "base_velocity") -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)

    commands = env.command_manager.get_command(command_name)
    # Scale by linear velocity only — ang_vel_z is excluded so that a pure rotation
    # command does not inject a gait phase signal, which would otherwise teach the
    # robot to march in place when only asked to turn.
    vel_magnitude = torch.norm(commands[:, :2], dim=1, keepdim=True).clamp(max=1.0)
    return phase * vel_magnitude
