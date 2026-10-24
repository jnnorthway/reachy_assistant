"""Awake pose for jump scare and wake, held by the movement loop."""

import time
import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest

from reachy_mini.reachy_mini import INIT_HEAD_POSE, SLEEP_HEAD_POSE
import reachy_mini_conversation_app.huggingface_realtime as hf_mod
from reachy_mini_conversation_app.moves import MovementManager
from reachy_mini_conversation_app.sleep_mode import (
    SPRING_UP_S,
    WAKE_POSE_S,
    AWAKE_ANTENNAS,
    AwakePoseMove,
    sleep_mode,
    command_awake_pose,
)
from reachy_mini_conversation_app.tools.wake_up import run_wake_motion
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.huggingface_realtime import HuggingFaceRealtimeHandler


def _sleeping_robot() -> MagicMock:
    robot = MagicMock()
    robot.get_current_head_pose.return_value = SLEEP_HEAD_POSE.copy()
    robot.get_current_joint_positions.return_value = ([0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-3.05, 3.05])
    return robot


def test_spring_uses_fast_pose_not_sdk_wake(monkeypatch: pytest.MonkeyPatch) -> None:
    """The scare rise is a short movement-loop goto, not the SDK wake animation."""
    monkeypatch.setattr("reachy_mini_conversation_app.sleep_mode.time.sleep", lambda _seconds: None)
    robot = _sleeping_robot()
    manager = MagicMock()
    manager._thread = None
    command_awake_pose(robot, manager, SPRING_UP_S)
    robot.wake_up.assert_not_called()
    robot.goto_target.assert_not_called()
    move = manager.play_now.call_args.args[0]
    assert isinstance(move, AwakePoseMove)
    assert move.duration == pytest.approx(SPRING_UP_S)
    assert 0.25 <= move.duration <= 0.45
    head, antennas, yaw = move.evaluate(move.duration)
    np.testing.assert_allclose(np.asarray(head), INIT_HEAD_POSE, atol=1e-5)
    assert yaw == pytest.approx(0.0)
    np.testing.assert_allclose(np.asarray(antennas), np.asarray(AWAKE_ANTENNAS), atol=1e-5)
    start_head, _, start_yaw = move.evaluate(0.0)
    np.testing.assert_allclose(np.asarray(start_head), SLEEP_HEAD_POSE, atol=1e-3)
    assert start_yaw == pytest.approx(0.4)
    installed_head, installed_antennas, installed_yaw = manager.install_pose.call_args.args
    np.testing.assert_allclose(installed_head, INIT_HEAD_POSE, atol=1e-5)
    assert installed_antennas == pytest.approx(AWAKE_ANTENNAS)
    assert installed_yaw == pytest.approx(0.0)
    manager.start.assert_called_once()


def test_wake_commands_upright_pose_and_clears_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wake leaves the upright pose commanded and does not fall back to sleep."""
    monkeypatch.setattr("reachy_mini_conversation_app.sleep_mode.time.sleep", lambda _seconds: None)
    sleep_mode.reset()
    sleep_mode.set_notifier(None)
    robot = _sleeping_robot()
    manager = MagicMock()
    manager._thread = None
    deps = ToolDependencies(reachy_mini=robot, movement_manager=manager)
    sleep_mode.enter_jump_scare()
    try:
        result = run_wake_motion(deps, "user asked to wake")
        assert result == {"status": "success", "action": "wake_up", "reason": "user asked to wake"}
        assert not sleep_mode.quiet()
        assert not sleep_mode.in_jump_scare()
        robot.wake_up.assert_not_called()
        robot.goto_target.assert_not_called()
        robot.goto_sleep.assert_not_called()
        move = manager.play_now.call_args.args[0]
        assert move.duration == pytest.approx(WAKE_POSE_S)
        head, _, yaw = move.evaluate(move.duration)
        np.testing.assert_allclose(np.asarray(head), INIT_HEAD_POSE, atol=1e-5)
        assert yaw == pytest.approx(0.0)
        robot.enable_wobbling.assert_called_once()
    finally:
        sleep_mode.reset()
        sleep_mode.set_notifier(None)


def test_installed_awake_pose_is_held_against_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """The control loop keeps the upright pose while the measured head is still asleep."""
    clock = {"t": 1000.0}
    monkeypatch.setattr("reachy_mini_conversation_app.moves.time.monotonic", lambda: clock["t"])
    robot = _sleeping_robot()
    manager = MovementManager(robot)
    manager.state.last_activity_time = 0.0
    manager.state.last_primary_pose = (SLEEP_HEAD_POSE.copy(), (-3.05, 3.05), 0.4)
    manager._head_tracking = True
    manager._track_anchor = SLEEP_HEAD_POSE.copy()
    manager._handle_command("install_pose", (np.eye(4), AWAKE_ANTENNAS, 0.0), clock["t"])
    manager._update_primary_motion(clock["t"])
    clock["t"] = 1000.2
    manager._update_primary_motion(clock["t"])
    head, antennas, yaw = manager._get_primary_pose(clock["t"])
    assert manager.state.current_move is None
    assert manager._track_anchor is None
    assert not manager._head_tracking
    np.testing.assert_allclose(head, INIT_HEAD_POSE, atol=1e-5)
    assert yaw == pytest.approx(0.0)
    assert antennas == pytest.approx(AWAKE_ANTENNAS)
    robot.stop_head_tracking.assert_called_once()


def test_play_now_replaces_held_sleep_with_the_spring(monkeypatch: pytest.MonkeyPatch) -> None:
    """A spring move carries the head off the held sleep pose."""
    clock = {"t": 500.0}
    monkeypatch.setattr("reachy_mini_conversation_app.moves.time.monotonic", lambda: clock["t"])
    robot = _sleeping_robot()
    manager = MovementManager(robot)
    manager.state.last_primary_pose = (SLEEP_HEAD_POSE.copy(), (-3.05, 3.05), 0.4)
    move = AwakePoseMove(SLEEP_HEAD_POSE.copy(), (-3.05, 3.05), 0.4, SPRING_UP_S)
    manager._handle_command("play_now", move, clock["t"])
    manager._update_primary_motion(clock["t"])
    assert manager.state.current_move is move
    manager.state.move_start_time = clock["t"] - (SPRING_UP_S * 0.5)
    head, _, yaw = manager._get_primary_pose(clock["t"])
    assert not np.allclose(np.asarray(head), SLEEP_HEAD_POSE, atol=1e-3)
    assert yaw == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_exact_scare_waits_for_playback_after_response_done(monkeypatch: pytest.MonkeyPatch) -> None:
    """response.done is not enough; the scare waits until playback has been asked to finish."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    order: list[str] = []

    async def create(**_kwargs: object) -> None:
        order.append("response.done")
        handler._exact_speech_done.set()

    async def wait_audio() -> None:
        order.append("playback")

    monkeypatch.setattr(handler, "_safe_response_create", create)
    monkeypatch.setattr(handler, "_wait_for_played_audio", wait_audio)
    try:
        await handler._speak_exact_scare()
    finally:
        sleep_mode.set_notifier(None)
    assert order == ["response.done", "playback"]


@pytest.mark.asyncio
async def test_paced_playback_holds_sleep_until_audio_ends(monkeypatch: pytest.MonkeyPatch) -> None:
    """A finished response still waits out scheduled audio and frames not yet played."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.paced_playback = True
    handler._playback_interrupted = False
    handler.playback_pending_frames = 1
    handler.playback_until = time.monotonic() + 30.0
    handler._exact_audio_samples = handler.SAMPLE_RATE
    handler._exact_audio_started_at = time.monotonic()
    slept: list[float] = []

    async def fake_sleep(delay: float) -> None:
        slept.append(delay)
        handler.playback_pending_frames = 0
        handler.playback_until = time.monotonic() - 1.0
        handler._exact_audio_samples = 0
        handler._exact_audio_started_at = time.monotonic() - 5.0

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    try:
        await handler._wait_for_played_audio()
    finally:
        sleep_mode.set_notifier(None)
    assert slept


@pytest.mark.asyncio
async def test_unpaced_playback_waits_out_queued_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a playout clock, sleep waits for the audio already queued."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.paced_playback = False
    handler._exact_audio_samples = handler.SAMPLE_RATE // 2
    handler.playback_until = time.monotonic() + 30.0
    handler.playback_pending_frames = 4
    slept: list[float] = []

    async def fake_sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    try:
        await handler._wait_for_played_audio()
    finally:
        sleep_mode.set_notifier(None)
    assert slept == [pytest.approx(0.5)]


@pytest.mark.asyncio
async def test_jump_scare_sleeps_only_after_the_line_finishes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The head springs, the yell plays out, and only then does he go back to sleep."""
    order: list[object] = []
    robot = MagicMock()
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=robot, movement_manager=MagicMock()))

    def spring(_robot: object, _manager: object, duration: float) -> None:
        order.append(duration)
        robot.wake_up.assert_not_called()

    async def create(**_kwargs: object) -> None:
        handler._exact_audio_samples = handler.SAMPLE_RATE
        handler._exact_audio_started_at = time.monotonic()
        handler.paced_playback = True
        handler.playback_pending_frames = 1
        handler.playback_until = time.monotonic() + 30.0
        assert handler._exact_speech_done is not None
        handler._exact_speech_done.set()

    async def fake_sleep(_delay: float) -> None:
        order.append("playing")
        assert sleep_mode.scare_in_progress()
        assert not sleep_mode.begin_scare()
        handler.playback_pending_frames = 0
        handler.playback_until = time.monotonic() - 1.0
        handler._exact_audio_samples = 0
        handler._exact_audio_started_at = time.monotonic() - 5.0

    def sleep_pose(_robot: object, _manager: object) -> None:
        order.append("sleep")
        assert "playing" in order

    monkeypatch.setattr(hf_mod, "command_awake_pose", spring)
    monkeypatch.setattr(hf_mod, "play_sleep_pose", sleep_pose)
    monkeypatch.setattr(handler, "_safe_response_create", create)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    sleep_mode.reset()
    sleep_mode.set_notifier(None)
    sleep_mode.enter_jump_scare()
    try:
        await handler._run_jump_scare()
        assert order[0] == pytest.approx(SPRING_UP_S)
        assert "playing" in order
        assert order[-1] == "sleep"
        assert order.index("playing") < order.index("sleep")
        robot.wake_up.assert_not_called()
        robot.goto_target.assert_not_called()
        assert sleep_mode.in_jump_scare()
        assert not sleep_mode.scare_in_progress()
    finally:
        sleep_mode.reset()
        sleep_mode.set_notifier(None)
