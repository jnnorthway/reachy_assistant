"""Tests for looking before the camera frame is taken."""

import math
import base64
from unittest.mock import MagicMock

import pytest

from reachy_mini.utils import create_head_pose
import reachy_mini_conversation_app.tools.look_direction as look_direction_mod
from reachy_mini_conversation_app.tools.move_head import MoveHead, AlignedLookMove
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.look_direction import LOOK_SETTLE_S, LookDirection


def _deps(*, camera_enabled: bool = True) -> ToolDependencies:
    reachy_mini = MagicMock()
    reachy_mini.get_current_joint_positions.return_value = ([0.05, 0, 0, 0, 0, 0, 0], (0.1, 0.2))
    reachy_mini.get_current_head_pose.return_value = create_head_pose(yaw=math.degrees(0.05))
    reachy_mini.media.get_frame_jpeg.return_value = b"\xff\xd8jpeg\xff\xd9"
    return ToolDependencies(
        reachy_mini=reachy_mini,
        movement_manager=MagicMock(),
        camera_enabled=camera_enabled,
    )


def _queued(deps: ToolDependencies) -> AlignedLookMove:
    queued = deps.movement_manager.queue_move.call_args.args[0]
    assert isinstance(queued, AlignedLookMove)
    return queued


@pytest.mark.asyncio
async def test_look_direction_left_waits_then_captures_with_body_yaw(monkeypatch: pytest.MonkeyPatch) -> None:
    """Left uses move_head's body yaw, and the frame is taken only after that motion plus settle."""
    deps = _deps()
    order: list[str] = []
    slept: list[float] = []

    async def fake_sleep(delay: float) -> None:
        order.append("sleep")
        slept.append(delay)

    def grab_frame() -> bytes:
        order.append("frame")
        return b"\xff\xd8jpeg\xff\xd9"

    monkeypatch.setattr(look_direction_mod.asyncio, "sleep", fake_sleep)
    deps.reachy_mini.media.get_frame_jpeg.side_effect = grab_frame

    head_deps = _deps()
    await MoveHead()(head_deps, direction="left")
    move_head_yaw = _queued(head_deps).target_body_yaw

    result = await LookDirection()(deps, direction="left")

    assert move_head_yaw != 0.0
    queued = _queued(deps)
    assert queued.target_body_yaw == pytest.approx(move_head_yaw)
    wait_s = deps.motion_duration_s + LOOK_SETTLE_S
    assert slept == [pytest.approx(wait_s)]
    assert order == ["sleep", "frame"]
    assert [name for name, _, _ in deps.movement_manager.method_calls[:2]] == ["clear_move_queue", "queue_move"]
    assert queued.duration > wait_s
    _, _, body_yaw_at_capture = queued.evaluate(wait_s)
    assert body_yaw_at_capture == pytest.approx(queued.target_body_yaw)
    assert result["b64_im"] == base64.b64encode(b"\xff\xd8jpeg\xff\xd9").decode("utf-8")
    assert result["direction"] == "left"
    assert result["captured_after_look"] is True
    assert "this picture" in result["note"]
    deps.reachy_mini.media.get_frame_jpeg.assert_called_once()


@pytest.mark.asyncio
async def test_look_direction_up_keeps_current_body_yaw(monkeypatch: pytest.MonkeyPatch) -> None:
    """Up tilts the head and leaves the base yaw where it was."""
    deps = _deps()

    async def fake_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(look_direction_mod.asyncio, "sleep", fake_sleep)
    await LookDirection()(deps, direction="up")

    queued = _queued(deps)
    assert queued.target_body_yaw == pytest.approx(0.05)
    _, _, body_yaw = queued.evaluate(queued.duration)
    assert body_yaw == pytest.approx(0.05)


@pytest.mark.asyncio
async def test_look_direction_rejects_unknown_direction() -> None:
    """An unknown direction is an error and does not move the robot."""
    deps = _deps()
    result = await LookDirection()(deps, direction="behind")
    assert result == {"error": "direction must be one of: left, right, up, down, front"}
    deps.movement_manager.queue_move.assert_not_called()
    deps.reachy_mini.media.get_frame_jpeg.assert_not_called()


@pytest.mark.asyncio
async def test_look_direction_camera_disabled_does_not_move_or_capture() -> None:
    """A disabled camera returns an error before any motion or frame read."""
    deps = _deps(camera_enabled=False)
    result = await LookDirection()(deps, direction="left")
    assert result == {"error": "Camera is disabled"}
    deps.movement_manager.queue_move.assert_not_called()
    deps.reachy_mini.media.get_frame_jpeg.assert_not_called()
