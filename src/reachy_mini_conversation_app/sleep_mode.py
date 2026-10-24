"""Sticky sleep and jump-scare phrase checks for the running conversation."""

import os
import re
import json
import time
import logging
import threading
from typing import Any, Literal
from pathlib import Path
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from platformdirs import user_config_dir

from reachy_mini import ReachyMini
from reachy_mini.motion.move import Move
from reachy_mini.reachy_mini import INIT_HEAD_POSE, INIT_ANTENNAS_JOINT_POSITIONS
from reachy_mini.utils.interpolation import (
    InterpolationTechnique,
    time_trajectory,
    linear_pose_interpolation,
)
from reachy_mini_conversation_app.config import INSTANCE_PATH_ENV, config


logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-z0-9']+")
_STATE_NAME = "sleep_state.json"

SCARE_LINE_ONE = "BOO!"
SCARE_LINE_TWO = "Ha gotcha!"
SCARE_INSTRUCTIONS = "Your entire spoken reply must be exactly these two sentences and nothing else: BOO! Ha gotcha!"

# Travel time of the jump. The SDK wake_up rise is about two seconds plus a roll.
SPRING_UP_S = 0.35
WAKE_POSE_S = 0.8
AWAKE_ANTENNAS: tuple[float, float] = (
    float(INIT_ANTENNAS_JOINT_POSITIONS[0]),
    float(INIT_ANTENNAS_JOINT_POSITIONS[1]),
)

QuietAction = Literal["respond", "ignore", "scare", "wake", "defer_wake"]


def _words(transcript: str) -> list[str]:
    return _WORD_RE.findall(transcript.lower())


def contains_word(transcript: str, word: str) -> bool:
    """Return whether transcript contains word as a whole word."""
    return word.lower() in _words(transcript)


def is_wake_request(transcript: str) -> bool:
    """Return whether transcript asks Kevin to wake up."""
    words = _words(transcript)
    return any(left == "wake" and right == "up" for left, right in zip(words, words[1:]))


def quiet_transcript_action(
    transcript: str,
    *,
    asleep: bool,
    jump_scare: bool,
    scare_running: bool,
) -> QuietAction:
    """Choose how a finished transcript is handled while sleep or jump scare is active."""
    if scare_running:
        if is_wake_request(transcript):
            return "defer_wake"
        return "ignore"
    quiet = asleep or jump_scare
    if quiet and is_wake_request(transcript):
        return "wake"
    if jump_scare and contains_word(transcript, "elephant"):
        return "scare"
    if quiet:
        return "ignore"
    return "respond"


def exact_scare_response() -> dict[str, Any]:
    """Return a response.create body that speaks the scare lines verbatim."""
    script = f"{SCARE_LINE_ONE} {SCARE_LINE_TWO}"
    return {
        "conversation": "none",
        "tool_choice": "none",
        "metadata": {"purpose": "exact_speech"},
        "instructions": SCARE_INSTRUCTIONS,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": script}],
            }
        ],
    }


def sleep_state_path() -> Path:
    """Return the on-disk sticky-sleep file for this instance."""
    if config.INSTANCE_PATH is not None:
        root = Path(config.INSTANCE_PATH)
    else:
        override = os.getenv(INSTANCE_PATH_ENV)
        root = Path(override).expanduser() if override else Path(user_config_dir("reachy_mini")) / "conversation_app"
    return root / _STATE_NAME


class SleepMode:
    """In-process sleep and jump-scare flags, optionally saved across restarts."""

    def __init__(self) -> None:
        """Start awake with no scare armed."""
        self._lock = threading.Lock()
        self._path: Path | None = None
        self._notifier: Callable[[], None] | None = None
        self._asleep = False
        self._jump_scare = False
        self._scare_running = False
        self._wake_after_scare = False

    def configure(self, path: Path) -> None:
        """Remember where sticky sleep is saved."""
        self._path = path

    def set_notifier(self, notifier: Callable[[], None] | None) -> None:
        """Register the handler callback invoked when sleep state changes."""
        self._notifier = notifier

    def clear_notifier(self, notifier: Callable[[], None]) -> None:
        """Drop the callback if it is still the active one."""
        if self._notifier is notifier:
            self._notifier = None

    def reset(self) -> None:
        """Clear in-memory flags without writing the save file."""
        with self._lock:
            self._asleep = False
            self._jump_scare = False
            self._scare_running = False
            self._wake_after_scare = False

    def snapshot(self) -> tuple[bool, bool, bool]:
        """Return asleep, jump-scare, and scare-running flags."""
        with self._lock:
            return self._asleep, self._jump_scare, self._scare_running

    def quiet(self) -> bool:
        """Return whether conversation and idle motion should stay suppressed."""
        with self._lock:
            return self._asleep or self._jump_scare

    def in_jump_scare(self) -> bool:
        """Return whether jump scare mode is armed."""
        with self._lock:
            return self._jump_scare

    def scare_in_progress(self) -> bool:
        """Return whether the scare motion and lines are already running."""
        with self._lock:
            return self._scare_running

    def mark_asleep(self) -> None:
        """Stay asleep across the running conversation until a wake request."""
        with self._lock:
            self._asleep = True
        self._persist()
        self._notify()
        logger.info("Sticky sleep on")

    def enter_jump_scare(self) -> None:
        """Sleep, listen, and arm the elephant scare until a wake phrase."""
        with self._lock:
            self._asleep = True
            self._jump_scare = True
        self._persist()
        self._notify()
        logger.info("Jump scare mode on")

    def wake(self) -> None:
        """Clear sticky sleep and jump scare."""
        with self._lock:
            self._asleep = False
            self._jump_scare = False
            self._scare_running = False
            self._wake_after_scare = False
        self._persist()
        self._notify()
        logger.info("Sticky sleep cleared")

    def begin_scare(self) -> bool:
        """Return False when a scare is already running or jump scare mode is off."""
        with self._lock:
            if not self._jump_scare or self._scare_running:
                return False
            self._scare_running = True
            return True

    def end_scare(self) -> None:
        """Allow another elephant trigger after the current scare finishes."""
        with self._lock:
            self._scare_running = False

    def note_wake_after_scare(self) -> None:
        """Remember a wake phrase spoken while the scare animation is running."""
        with self._lock:
            self._wake_after_scare = True

    def take_wake_after_scare(self) -> bool:
        """Return and clear a wake phrase that arrived during a scare."""
        with self._lock:
            wake = self._wake_after_scare
            self._wake_after_scare = False
            return wake

    def load(self) -> None:
        """Restore sticky sleep from disk when a save file exists."""
        path = self._path
        if path is None or not path.is_file():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read sleep state: %s", exc)
            return
        if not isinstance(payload, dict):
            logger.warning("Ignoring malformed sleep state")
            return
        with self._lock:
            self._asleep = bool(payload.get("asleep"))
            self._jump_scare = bool(payload.get("jump_scare"))
            self._scare_running = False
            self._wake_after_scare = False
        logger.info("Loaded sleep state asleep=%s jump_scare=%s", self._asleep, self._jump_scare)

    def _persist(self) -> None:
        path = self._path
        if path is None:
            return
        with self._lock:
            payload = {"asleep": self._asleep, "jump_scare": self._jump_scare}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_text(json.dumps(payload), encoding="utf-8")
            temporary.replace(path)
        except OSError as exc:
            logger.warning("Could not save sleep state: %s", exc)

    def _notify(self) -> None:
        notifier = self._notifier
        if notifier is None:
            return
        try:
            notifier()
        except Exception as exc:
            logger.warning("Sleep state notifier failed: %s", exc)


sleep_mode = SleepMode()


def play_sleep_pose(robot: Any, movement_manager: Any) -> str | None:
    """Stop idle motion and move to the sleep pose."""
    try:
        robot.disable_wobbling()
    except Exception as exc:
        logger.debug("Error disabling wobbling before sleep: %s", exc)
    try:
        set_head_tracking = getattr(movement_manager, "set_head_tracking", None)
        if set_head_tracking is not None:
            set_head_tracking(False)
        movement_manager.clear_move_queue()
        movement_manager.stop(reset_to_neutral=False)
    except Exception as exc:
        logger.error("Failed to stop movement before sleep: %s", exc)
    try:
        robot.goto_sleep()
    except Exception as exc:
        logger.error("Failed to move Reachy Mini to sleep pose: %s", exc)
        return f"{type(exc).__name__}: {exc}"
    return None


def hold_sleep(robot: Any, movement_manager: Any) -> dict[str, Any]:
    """Move to the sleep pose and keep the conversation listening."""
    logger.info("Holding sleep pose; the conversation keeps listening.")
    sleep_mode.mark_asleep()
    error = play_sleep_pose(robot, movement_manager)
    result: dict[str, Any] = {
        "status": "sleeping" if error is None else "sleep_pose_failed",
        "listening": True,
    }
    if error is not None:
        result["error"] = f"go_to_sleep movement failed: {error}"
    return result


class AwakePoseMove(Move):  # type: ignore[misc]
    """Rise from the current pose to the upright awake pose."""

    def __init__(
        self,
        start_head: NDArray[np.float64],
        start_antennas: tuple[float, float],
        start_body_yaw: float,
        duration: float,
    ) -> None:
        """Store the pose to leave and how long the rise takes."""
        self._start_head = np.array(start_head, dtype=np.float64, copy=True)
        self._target_head = np.array(INIT_HEAD_POSE, dtype=np.float64, copy=True)
        self._start_antennas = (float(start_antennas[0]), float(start_antennas[1]))
        self._target_antennas = AWAKE_ANTENNAS
        self._start_body_yaw = float(start_body_yaw)
        self._duration = duration

    @property
    def duration(self) -> float:
        """Duration property required by the Move interface."""
        return self._duration

    def evaluate(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        """Sample the rise; the sample at the end is the upright pose."""
        if self._duration <= 0.0:
            raw = 1.0
        else:
            raw = min(1.0, max(0.0, t / self._duration))
        progress = time_trajectory(raw, method=InterpolationTechnique.MIN_JERK)
        head = linear_pose_interpolation(self._start_head, self._target_head, progress, yaw_as_scalar=True)
        antennas = np.array(
            [
                self._start_antennas[0] + (self._target_antennas[0] - self._start_antennas[0]) * progress,
                self._start_antennas[1] + (self._target_antennas[1] - self._start_antennas[1]) * progress,
            ],
            dtype=np.float64,
        )
        body_yaw = self._start_body_yaw * (1.0 - progress)
        return head, antennas, body_yaw


def command_awake_pose(robot: Any, movement_manager: Any, duration: float) -> None:
    """Enable the motors and leave the upright pose commanded by the movement loop."""
    if duration <= 0.0:
        raise ValueError("awake pose duration must be positive")
    try:
        robot.disable_wobbling()
    except Exception as exc:
        logger.debug("Error disabling wobbling before awake pose: %s", exc)
    try:
        robot.enable_motors()
    except Exception as exc:
        logger.error("Could not enable motors for the awake pose: %s", exc)
        raise
    try:
        head_joints, antennas = robot.get_current_joint_positions()
        current_head = robot.get_current_head_pose()
        start_yaw = float(head_joints[0])
        start_antennas = (float(antennas[0]), float(antennas[1]))
    except Exception as exc:
        logger.error("Could not read the pose for the awake move: %s", exc)
        raise
    move = AwakePoseMove(
        start_head=np.asarray(current_head, dtype=np.float64),
        start_antennas=start_antennas,
        start_body_yaw=start_yaw,
        duration=duration,
    )
    logger.info("Moving to awake pose in %.2fs", duration)
    movement_manager.play_now(move)
    thread = getattr(movement_manager, "_thread", None)
    alive = getattr(thread, "is_alive", None)
    if thread is None or not callable(alive) or not alive():
        movement_manager.start()
    time.sleep(duration)
    movement_manager.install_pose(
        np.array(INIT_HEAD_POSE, dtype=np.float64, copy=True),
        AWAKE_ANTENNAS,
        0.0,
    )


def apply_service_prestart() -> None:
    """Enable motors, and wake only when sticky sleep is not saved."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sleep_mode.configure(sleep_state_path())
    sleep_mode.load()
    robot = ReachyMini()
    try:
        robot.enable_motors()
    except Exception as exc:
        logger.error("Prestart could not enable motors: %s", exc)
    if sleep_mode.quiet():
        logger.info("Sticky sleep is active; skipping wake animation.")
        try:
            robot.goto_sleep()
        except Exception as exc:
            logger.error("Prestart sleep pose failed: %s", exc)
        return
    try:
        robot.wake_up()
    except Exception as exc:
        logger.error("Prestart wake failed: %s", exc)
