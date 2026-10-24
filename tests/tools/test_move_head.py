import math
from typing import Any
from unittest.mock import MagicMock

import pytest

from reachy_mini.utils import create_head_pose
from reachy_mini.utils.rotation import Rotation as R
from reachy_mini_conversation_app.tools.move_head import MoveHead, AlignedLookMove
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def _deps() -> ToolDependencies:
    reachy_mini = MagicMock()
    # get_current_joint_positions() -> (head_joints[7], antenna_joints[2]); body_yaw is
    # head_joints[0]. Distinct values below (0.05 vs 0.1/0.2) so a test can tell which
    # tuple start_body_yaw actually came from.
    reachy_mini.get_current_joint_positions.return_value = ([0.05, 0, 0, 0, 0, 0, 0], (0.1, 0.2))
    reachy_mini.get_current_head_pose.return_value = create_head_pose(yaw=math.degrees(0.05))
    return ToolDependencies(reachy_mini=reachy_mini, movement_manager=MagicMock())


def _queued(deps: ToolDependencies) -> AlignedLookMove:
    queued = deps.movement_manager.queue_move.call_args.args[0]
    assert isinstance(queued, AlignedLookMove)
    return queued


def _head_yaw(pose: Any) -> float:
    return float(R.from_matrix(pose[:3, :3]).as_euler("xyz")[2])


@pytest.mark.asyncio
async def test_move_head_rejects_non_string_direction() -> None:
    """A non-string direction is rejected without touching the robot."""
    result = await MoveHead()(_deps(), direction=42)
    assert result == {"error": "direction must be a string"}


@pytest.mark.asyncio
async def test_move_head_queues_goto_move() -> None:
    """A valid direction queues one aligned look and marks the robot moving."""
    deps = _deps()
    result = await MoveHead()(deps, direction="left")
    assert result == {"status": "looking left"}
    queued_move = _queued(deps)
    assert isinstance(queued_move, AlignedLookMove)
    assert deps.movement_manager.queue_move.call_count == 1
    deps.movement_manager.set_moving_state.assert_called_once_with(deps.motion_duration_s)


@pytest.mark.asyncio
async def test_move_head_reads_body_yaw_from_head_joints() -> None:
    """start_body_yaw comes from the head-joints tuple, not the antenna-joints tuple."""
    deps = _deps()
    await MoveHead()(deps, direction="left")
    queued_move = _queued(deps)
    assert queued_move.start_body_yaw == 0.05


@pytest.mark.asyncio
async def test_move_head_left_turns_the_base() -> None:
    """Looking left yaws the base and aims the head with it."""
    deps = _deps()
    await MoveHead()(deps, direction="left")
    queued_move = _queued(deps)
    assert queued_move.target_body_yaw == pytest.approx(math.radians(50))
    for progress in (0.0, 0.25, 0.5, 0.75, 1.0):
        head, _, body_yaw = queued_move.evaluate(progress * queued_move.duration)
        assert head is not None
        assert _head_yaw(head) == pytest.approx(body_yaw)
    _, _, yaw_quarter = queued_move.evaluate(0.25 * queued_move.duration)
    linear_quarter = queued_move.start_body_yaw + 0.25 * (queued_move.target_body_yaw - queued_move.start_body_yaw)
    assert abs(yaw_quarter - queued_move.start_body_yaw) < abs(linear_quarter - queued_move.start_body_yaw)


@pytest.mark.asyncio
async def test_move_head_up_keeps_the_current_base_angle() -> None:
    """Tilting the head does not snap the base back to center."""
    deps = _deps()
    await MoveHead()(deps, direction="up")
    queued_move = _queued(deps)
    assert queued_move.target_body_yaw == pytest.approx(0.05)


@pytest.mark.asyncio
async def test_move_head_front_centers_the_base() -> None:
    """Looking front returns the base yaw to zero."""
    deps = _deps()
    await MoveHead()(deps, direction="front")
    queued_move = _queued(deps)
    assert queued_move.target_body_yaw == 0.0


@pytest.mark.asyncio
async def test_move_head_reports_robot_failure() -> None:
    """A robot read failure is returned as an error, not raised into the loop."""
    deps = _deps()
    deps.reachy_mini.get_current_head_pose.side_effect = RuntimeError("boom")
    result = await MoveHead()(deps, direction="up")
    assert "error" in result
    assert "RuntimeError" in result["error"]
