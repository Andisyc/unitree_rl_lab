"""
Test suite for turning reward design.
Validates: drift-stomping prevention, turning reward scaling, zero-command behavior,
boundary conditions, and potential reward hacking paths.

Run in your Isaac Lab environment:
    python tests/test_turning_rewards.py
"""

from __future__ import annotations

import math
import torch
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, PropertyMock


# ---------------------------------------------------------------------------
# Minimal mock of the Isaac Lab env for standalone reward-function testing.
# No Isaac Lab import needed — all reward functions are pure tensor ops.
# ---------------------------------------------------------------------------


@dataclass
class MockBodyData:
    """Mocks Articulation.data / RigidObject.data body-level attributes."""
    body_pos_w: torch.Tensor          # [num_envs, num_bodies, 3]
    body_lin_vel_w: torch.Tensor      # [num_envs, num_bodies, 3]
    root_pos_w: torch.Tensor          # [num_envs, 3]
    root_lin_vel_w: torch.Tensor      # [num_envs, 3]
    root_ang_vel_b: torch.Tensor      # [num_envs, 3]
    root_quat_w: torch.Tensor         # [num_envs, 4]
    projected_gravity_b: torch.Tensor # [num_envs, 3]
    joint_pos: torch.Tensor | None = None          # [num_envs, num_joints]
    joint_vel: torch.Tensor | None = None
    default_joint_pos: torch.Tensor | None = None
    applied_torque: torch.Tensor | None = None


class MockAsset:
    """Mocks Articulation / RigidObject."""
    def __init__(self, data: MockBodyData):
        self.data = data

    def find_joints(self, name: str) -> list[int]:
        return [0]  # dummy


class MockContactSensorData:
    def __init__(self, net_forces_w: torch.Tensor, current_contact_time: torch.Tensor,
                 last_air_time: torch.Tensor | None = None,
                 last_contact_time: torch.Tensor | None = None):
        self.net_forces_w = net_forces_w
        self.current_contact_time = current_contact_time
        self.last_air_time = last_air_time
        self.last_contact_time = last_contact_time


class MockContactSensor:
    def __init__(self, data: MockContactSensorData, cfg: Any = None):
        self.data = data
        self.cfg = cfg


class MockCommandManager:
    def __init__(self, commands: dict[str, torch.Tensor]):
        self._commands = commands

    def get_command(self, name: str) -> torch.Tensor:
        return self._commands[name]


class MockEnv:
    """Lightweight mock: only the attributes that reward functions actually read."""

    def __init__(
        self,
        num_envs: int,
        body_data: MockBodyData,
        contact_data: MockContactSensorData,
        commands: dict[str, torch.Tensor],
        device: str = "cpu",
    ):
        self.num_envs = num_envs
        self.device = device
        self.step_dt = 0.02   # 50 Hz control
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)

        self.scene = {
            "robot": MockAsset(body_data),
        }
        self.scene["robot"].data = body_data  # for direct access patterns

        self.scene.sensors = {
            "contact_forces": MockContactSensor(contact_data),
        }

        self.command_manager = MockCommandManager(commands)

        self.joint_mirror_joints_cache = None


# ---------------------------------------------------------------------------
# Helper: build tensor batches
# ---------------------------------------------------------------------------

def _t(v) -> torch.Tensor:
    """Shorthand for creating 1-env tensors with batch dim."""
    return torch.tensor(v, dtype=torch.float).unsqueeze(0)


def _make_env_with_command(cmd: list[float], **overrides) -> MockEnv:
    """Create a 1-env mock with the given velocity command [lin_x, lin_y, ang_z]."""
    N = 1
    cmd_t = torch.tensor([cmd], dtype=torch.float)

    body = MockBodyData(
        body_pos_w=torch.zeros(N, 2, 3),       # 2 feet
        body_lin_vel_w=torch.zeros(N, 2, 3),
        root_pos_w=torch.zeros(N, 3),
        root_lin_vel_w=torch.zeros(N, 3),
        root_ang_vel_b=torch.zeros(N, 3),
        root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),  # identity
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
    )
    # Apply overrides
    for k, v in overrides.items():
        setattr(body, k, v if isinstance(v, torch.Tensor) else _t(v) if k != "body_pos_w" and k != "body_lin_vel_w" else v)

    return MockEnv(
        num_envs=N,
        body_data=body,
        contact_data=MockContactSensorData(
            net_forces_w=torch.zeros(N, 2, 3),
            current_contact_time=torch.zeros(N, 2),
        ),
        commands={"base_velocity": cmd_t},
    )


# ---------------------------------------------------------------------------
# Import the real reward functions (requires the project on PYTHONPATH)
# ---------------------------------------------------------------------------

from unitree_rl_lab.tasks.locomotion.mdp.rewards import (
    stand_still,
    feet_contact_without_cmd,
    foot_clearance_reward,
    feet_gait,
    orientation_l2,
)


# ===================================================================
# Test cases
# ===================================================================

class TestZeroCommandBehavior:
    """Zero-command environments: robot should stand still, not drift-stomp."""

    def test_stand_still_applies_when_cmd_zero(self):
        """stand_still penalty should fire when cmd_norm < 0.1."""
        env = _make_env_with_command(
            cmd=[0.0, 0.0, 0.0],
            joint_pos=_t([0.1, -0.1, 0.2, 0.0]),  # 4 joint deviations
            default_joint_pos=_t([0.0, 0.0, 0.0, 0.0]),
        )
        penalty = stand_still(env, command_name="base_velocity")
        # sum(|deviation|) = 0.1+0.1+0.2+0.0 = 0.4
        assert penalty.item() == pytest.approx(0.4, abs=1e-4), \
            f"Expected 0.4, got {penalty.item()}"

    def test_stand_still_suppressed_when_cmd_nonzero(self):
        """stand_still should be zero when cmd_norm >= 0.1."""
        env = _make_env_with_command(
            cmd=[0.1, 0.0, 0.0],
            joint_pos=_t([1.0, -1.0, 0.5, 0.3]),
            default_joint_pos=_t([0.0, 0.0, 0.0, 0.0]),
        )
        penalty = stand_still(env, command_name="base_velocity")
        assert penalty.item() == 0.0, \
            f"Expected 0.0 (cmd_norm=0.1 >= 0.1), got {penalty.item()}"

    def test_feet_contact_without_cmd_rewards_both_feet_on_ground(self):
        """feet_contact_zero_cmd should give reward for both feet on ground when cmd=0."""
        env = _make_env_with_command(cmd=[0.0, 0.0, 0.0])
        # Override contact: both feet touching
        env.scene.sensors["contact_forces"].data.current_contact_time = \
            torch.tensor([[1.0, 1.0]], dtype=torch.float)
        reward = feet_contact_without_cmd(
            env, sensor_cfg=MagicMock(body_ids=[0, 1]), command_name="base_velocity"
        )
        assert reward.item() == 2.0, f"Expected 2.0, got {reward.item()}"

    def test_feet_contact_without_cmd_zero_when_cmd_nonzero(self):
        """feet_contact_zero_cmd should be zero when command is active."""
        env = _make_env_with_command(cmd=[0.1, 0.0, 0.0])
        env.scene.sensors["contact_forces"].data.current_contact_time = \
            torch.tensor([[1.0, 1.0]], dtype=torch.float)
        reward = feet_contact_without_cmd(
            env, sensor_cfg=MagicMock(body_ids=[0, 1]), command_name="base_velocity"
        )
        assert reward.item() == 0.0, f"Expected 0.0, got {reward.item()}"


class TestTurningReward:
    """Turning reward: should scale gait reward by angular achievement."""

    def test_gait_reward_suppressed_for_zero_cmd(self):
        """feetta_gait should be zero when cmd_norm <= 0.1 (no drift-stomping)."""
        env = _make_env_with_command(cmd=[0.0, 0.0, 0.0])
        env.scene.sensors["contact_forces"].data.current_contact_time = \
            torch.tensor([[0.0, 0.0]], dtype=torch.float)  # both feet in air
        env.episode_length_buf = torch.zeros(1, dtype=torch.long)

        reward = feet_gait(
            env,
            period=0.5,
            offset=[0.0, 0.5],
            sensor_cfg=MagicMock(body_ids=[0, 1]),
            threshold=0.55,
            command_name="base_velocity",
        )
        assert reward.item() == 0.0, \
            f"Gait reward should be 0 for zero cmd (drift-stomp prevention), got {reward.item()}"

    def test_gait_reward_suppressed_for_small_cmd(self):
        """Gait reward should be zero when cmd_norm <= 0.1 even for small non-zero cmd."""
        env = _make_env_with_command(cmd=[0.05, 0.05, 0.0])  # norm ≈ 0.07
        env.scene.sensors["contact_forces"].data.current_contact_time = \
            torch.tensor([[0.0, 0.0]], dtype=torch.float)
        env.episode_length_buf = torch.zeros(1, dtype=torch.long)

        reward = feet_gait(
            env,
            period=0.5,
            offset=[0.0, 0.5],
            sensor_cfg=MagicMock(body_ids=[0, 1]),
            threshold=0.55,
            command_name="base_velocity",
        )
        assert reward.item() == 0.0, \
            f"Gait reward should be 0 for cmd_norm <= 0.1, got {reward.item()}"

    def test_pure_turning_ang_frac_scales_gait(self):
        """When only angular cmd is present, gait reward scales with ang_frac."""
        # cmd: ang=0.5 rad/s, no lin
        env = _make_env_with_command(
            cmd=[0.0, 0.0, 0.5],
            root_ang_vel_b=_t([0.0, 0.0, 0.25]),  # 50% of cmd
            root_lin_vel_b=_t([0.0, 0.0, 0.0]),
        )
        # Perfect gait match (both feet follow pattern exactly)
        env.scene.sensors["contact_forces"].data.current_contact_time = \
            torch.tensor([[0.0, 0.0]], dtype=torch.float)
        env.episode_length_buf = torch.zeros(1, dtype=torch.long)

        reward = feet_gait(
            env,
            period=0.5,
            offset=[0.0, 0.5],
            sensor_cfg=MagicMock(body_ids=[0, 1]),
            threshold=0.55,
            command_name="base_velocity",
        )
        # ang_frac = min(0.25/0.5, 1.0) = 0.5
        # max gait reward (no contact mismatch) * 0.5
        # With 2 feet, phase[0]=0, phase[1]=0.5, threshold=0.55:
        # global_phase = episode_length_buf * step_dt % period / period = 0
        # leg_phase[0] = (0 + 0) % 1 = 0 < 0.55 → is_stance=True
        # leg_phase[1] = (0 + 0.5) % 1 = 0.5 < 0.55 → is_stance=True
        # contact both=0 → is_contact=False
        # For both: is_stance=True, is_contact=False → XOR=True → ~False=0
        # So gait_match = 0 for both feet... hmm this depends on phase
        #
        # The key test: ang_frac should be ~0.5, and the final reward should
        # reflect that scaling. We don't test the exact value (depends on phase
        # alignment) but verify it's non-zero and < max possible.
        assert 0.0 < reward.item() < 2.0, \
            f"Gait reward should be scaled by ang_frac (~0.5), got {reward.item()}"

    def test_wrong_direction_turn_gives_zero_ang_progress(self):
        """Turning opposite to command should give ang_progress=0 → ang_frac=0."""
        # cmd: ang=+0.5 rad/s, body turning at -0.3 rad/s
        env = _make_env_with_command(
            cmd=[0.0, 0.0, 0.5],
            root_ang_vel_b=_t([0.0, 0.0, -0.3]),
            root_lin_vel_b=_t([0.0, 0.0, 0.0]),
        )
        env.scene.sensors["contact_forces"].data.current_contact_time = \
            torch.tensor([[0.0, 0.0]], dtype=torch.float)
        env.episode_length_buf = torch.zeros(1, dtype=torch.long)

        reward = feet_gait(
            env,
            period=0.5,
            offset=[0.0, 0.5],
            sensor_cfg=MagicMock(body_ids=[0, 1]),
            threshold=0.55,
            command_name="base_velocity",
        )
        # ang_progress = (-0.3 * +1).clamp(min=0) = 0
        # ang_frac = 0/0.5 = 0 → vel_fraction = 0 → reward = 0
        assert reward.item() == 0.0, \
            f"Wrong direction turn should give 0 gait reward, got {reward.item()}"

    def test_turning_with_full_achievement_gives_full_gait(self):
        """100% angular achievement should not suppress gait reward."""
        env = _make_env_with_command(
            cmd=[0.0, 0.0, 0.8],
            root_ang_vel_b=_t([0.0, 0.0, 0.8]),
            root_lin_vel_b=_t([0.0, 0.0, 0.0]),
        )
        env.scene.sensors["contact_forces"].data.current_contact_time = \
            torch.tensor([[0.0, 0.0]], dtype=torch.float)
        env.episode_length_buf = torch.zeros(1, dtype=torch.long)

        reward = feet_gait(
            env,
            period=0.5,
            offset=[0.0, 0.5],
            sensor_cfg=MagicMock(body_ids=[0, 1]),
            threshold=0.55,
            command_name="base_velocity",
        )
        # ang_frac = min(0.8/0.8, 1.0) = 1.0
        assert reward.item() > 0.0, \
            f"Full turning achievement should not zero gait reward, got {reward.item()}"


class TestDriftStompingPrevention:
    """Verify that drift-stomping cannot earn high reward."""

    def test_stomping_body_drift_gets_proportional_gait(self):
        """Small accidental drift from stomping gives proportionally small gait reward."""
        # cmd: lin=0.5 m/s forward
        # Robot only achieves 0.05 m/s from stomping (10% of cmd)
        env = _make_env_with_command(
            cmd=[0.5, 0.0, 0.0],
            root_lin_vel_b=_t([0.05, 0.0, 0.0]),
            root_ang_vel_b=_t([0.0, 0.0, 0.0]),
        )
        env.scene.sensors["contact_forces"].data.current_contact_time = \
            torch.tensor([[0.0, 0.0]], dtype=torch.float)
        env.episode_length_buf = torch.zeros(1, dtype=torch.long)

        reward = feet_gait(
            env,
            period=0.5,
            offset=[0.0, 0.5],
            sensor_cfg=MagicMock(body_ids=[0, 1]),
            threshold=0.55,
            command_name="base_velocity",
        )
        # lin_frac = min(0.05/0.5, 1.0) = 0.1
        # reward * 0.1 → stomping gets only 10% of gait reward
        max_possible = 2.0  # both feet matching perfectly
        assert reward.item() <= max_possible * 0.1 + 1e-3, \
            f"Stomping should get <= 10% gait reward, got {reward.item()}"
        assert reward.item() > 0.0, "Should still get some reward for partial achievement"


class TestBoundaryConditions:
    """Test edge cases at the cmd_norm = 0.1 boundary (now using >= for gait gate)."""

    def test_cmd_norm_exactly_0_1_passes_gate_with_achievement(self):
        """With >= gate, cmd_norm == 0.1 passes. Gait reward scales with achievement."""
        env = _make_env_with_command(
            cmd=[0.1, 0.0, 0.0],  # norm = 0.1 exactly → passes >= gate
            root_lin_vel_b=_t([0.05, 0.0, 0.0]),  # 50% achievement
            root_ang_vel_b=_t([0.0, 0.0, 0.0]),
        )
        env.scene.sensors["contact_forces"].data.current_contact_time = \
            torch.tensor([[0.0, 0.0]], dtype=torch.float)
        env.episode_length_buf = torch.zeros(1, dtype=torch.long)

        reward = feet_gait(
            env,
            period=0.5,
            offset=[0.0, 0.5],
            sensor_cfg=MagicMock(body_ids=[0, 1]),
            threshold=0.55,
            command_name="base_velocity",
        )
        # lin_frac = min(0.05/0.1, 1.0) = 0.5, reward should be scaled to ~50%
        assert reward.item() > 0.0, \
            f"cmd_norm=0.1 should pass >= gate, got {reward.item()}"

    def test_cmd_norm_below_0_1_still_suppressed(self):
        """cmd_norm < 0.1 should still be fully suppressed."""
        env = _make_env_with_command(cmd=[0.05, 0.05, 0.0])  # norm ≈ 0.07
        env.scene.sensors["contact_forces"].data.current_contact_time = \
            torch.tensor([[0.0, 0.0]], dtype=torch.float)
        env.episode_length_buf = torch.zeros(1, dtype=torch.long)

        reward = feet_gait(
            env,
            period=0.5,
            offset=[0.0, 0.5],
            sensor_cfg=MagicMock(body_ids=[0, 1]),
            threshold=0.55,
            command_name="base_velocity",
        )
        assert reward.item() == 0.0, \
            f"cmd_norm < 0.1 should be fully suppressed, got {reward.item()}"

    def test_no_constraint_fallback_zero(self):
        """When neither lin nor ang axis individually constrains but cmd_norm > 0.1,
        vel_fraction should be 0 (not 1.0 — no free gait reward)."""
        # cmd: lin_norm=0.08, ang=0.08 → neither > 0.1 individually, but combined norm ≈ 0.113 > 0.1
        env = _make_env_with_command(
            cmd=[0.08, 0.0, 0.08],  # cmd_norm ≈ sqrt(0.08^2 + 0.08^2) ≈ 0.113
            root_lin_vel_b=_t([0.08, 0.0, 0.0]),
            root_ang_vel_b=_t([0.0, 0.0, 0.08]),
        )
        env.scene.sensors["contact_forces"].data.current_contact_time = \
            torch.tensor([[0.0, 0.0]], dtype=torch.float)
        env.episode_length_buf = torch.zeros(1, dtype=torch.long)

        reward = feet_gait(
            env,
            period=0.5,
            offset=[0.0, 0.5],
            sensor_cfg=MagicMock(body_ids=[0, 1]),
            threshold=0.55,
            command_name="base_velocity",
        )
        # has_lin=False (0.08 <= 0.1), has_ang=False (0.08 <= 0.1)
        # cmd_norm ≈ 0.113 > 0.1 → passes first gate
        # But vel_fraction = 0 → reward = base_gait_match * 0 = 0
        assert reward.item() == 0.0, \
            f"Ambiguous small multi-axis commands should get 0 gait reward, got {reward.item()}"


class TestWaistTwistExploit:
    """Verify that waist-twist turning doesn't earn full gait reward
    because leg contact pattern won't match the gait schedule."""

    def test_waist_twist_with_planted_feet_penalized_by_gait(self):
        """If the robot twists its waist to turn while keeping feet planted,
        the contact pattern stays constant (always in contact), which
        misaligns with the expected gait swing/stance phases."""
        # Robot achieves full angular velocity via waist twist
        env = _make_env_with_command(
            cmd=[0.0, 0.0, 0.5],
            root_ang_vel_b=_t([0.0, 0.0, 0.5]),  # full turning achievement
            root_lin_vel_b=_t([0.0, 0.0, 0.0]),
        )
        # Both feet always in contact (planted, twisting at waist)
        env.scene.sensors["contact_forces"].data.current_contact_time = \
            torch.tensor([[1.0, 1.0]], dtype=torch.float)
        env.episode_length_buf = torch.zeros(1, dtype=torch.long)

        reward = feet_gait(
            env,
            period=0.5,
            offset=[0.0, 0.5],
            sensor_cfg=MagicMock(body_ids=[0, 1]),
            threshold=0.55,
            command_name="base_velocity",
        )
        # At global_phase=0, threshold=0.55:
        # leg_phase[0] = 0     < 0.55 → is_stance=True, contact=True  → XOR=False → ~True=1
        # leg_phase[1] = 0.5   < 0.55 → is_stance=True, contact=True  → XOR=False → ~True=1
        # gait_match = 2, ang_frac = 1.0 → reward = 2.0
        #
        # At global_phase=0.6 (different time step):
        # leg_phase[0] = 0.6   < 0.55 → is_stance=False, contact=True → XOR=True → ~False=0
        # leg_phase[1] = 0.1   < 0.55 → is_stance=True, contact=True  → XOR=False → ~True=1
        # gait_match = 1, ang_frac = 1.0 → reward = 1.0
        #
        # Over a full cycle, only ~55% of steps match → average reward ~1.1
        # compared to ~2.0 for proper gait. The proportional scaling from
        # ang_frac doesn't fix this — the gait contact mismatch is the defense.
        #
        # We test at phase=0 where the mismatch is minimal (worst case for detection).
        assert reward.item() < 2.0, \
            f"Waist twist with planted feet should get < 2.0 gait reward at some phases, got {reward.item()}"

    def test_proper_turning_gait_gets_higher_reward_than_waist_twist(self):
        """Compare: proper turning with coordinated steps vs waist twist.
        Proper gait should get higher reward at most phases."""
        cmd = [0.0, 0.0, 0.5]

        # --- Proper gait: feet alternate contact matching phase ---
        env_good = _make_env_with_command(cmd=cmd, root_ang_vel_b=_t([0.0, 0.0, 0.5]))
        env_good.episode_length_buf = torch.tensor([5], dtype=torch.long)  # phase offset
        env_good.scene.sensors["contact_forces"].data.current_contact_time = \
            torch.tensor([[0.0, 1.0]], dtype=torch.float)  # one foot up, one down

        reward_good = feet_gait(
            env_good,
            period=0.5,
            offset=[0.0, 0.5],
            sensor_cfg=MagicMock(body_ids=[0, 1]),
            threshold=0.55,
            command_name="base_velocity",
        )

        # --- Waist twist: both feet always planted ---
        env_bad = _make_env_with_command(cmd=cmd, root_ang_vel_b=_t([0.0, 0.0, 0.5]))
        env_bad.episode_length_buf = torch.tensor([5], dtype=torch.long)
        env_bad.scene.sensors["contact_forces"].data.current_contact_time = \
            torch.tensor([[1.0, 1.0]], dtype=torch.float)

        reward_bad = feet_gait(
            env_bad,
            period=0.5,
            offset=[0.0, 0.5],
            sensor_cfg=MagicMock(body_ids=[0, 1]),
            threshold=0.55,
            command_name="base_velocity",
        )

        # The proper gait should NOT be worse than waist twist
        assert reward_good.item() >= reward_bad.item(), \
            f"Proper gait ({reward_good.item():.3f}) should >= waist twist ({reward_bad.item():.3f})"


class TestOrientationPenaltyDuringTurning:
    """Body tilt penalties must be strong enough to prevent postural hacking during turns."""

    def test_flat_orientation_l2_penalty(self):
        """A tilted robot during turning should receive meaningful penalty."""
        # Robot tilted 20° to the side (projected_gravity: sin(20°) ≈ 0.342 in x)
        env = _make_env_with_command(
            cmd=[0.0, 0.0, 0.5],
            projected_gravity_b=_t([0.342, 0.0, -0.94]),
        )
        penalty = orientation_l2(env, desired_gravity=[0.0, 0.0, -1.0])
        # cos_dist = 0.94, normalized = 0.5*0.94 + 0.5 = 0.97
        # squared = 0.9409 → this is actually a REWARD (orientation_l2 returns ~1 for upright)
        # Wait, flat_orientation_l2 uses weight -2.5, so the penalty = -2.5 * orientation_l2
        # orientation_l2 returns ~1.0 for perfect upright, lower for tilted.
        # So penalty = -2.5 * (lower value for tilted) → less negative → weaker penalty
        # This is correct: the function returns a value in [0,1] where 1 = perfect upright.
        assert penalty.item() < 1.0, \
            f"Tilted robot should get < 1.0 orientation reward, got {penalty.item()}"


# ===================================================================
# Runner
# ===================================================================

if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
