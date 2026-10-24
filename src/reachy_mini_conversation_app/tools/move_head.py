import math
import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from reachy_mini.utils import create_head_pose
from reachy_mini.motion.move import Move
from reachy_mini.utils.rotation import Rotation as R
from reachy_mini.utils.interpolation import InterpolationTechnique, time_trajectory
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Left and right yaw the base. Head yaw is rebuilt from that same angle at every
# sample so the neck stays square to the body for the whole turn. The base can go
# to about ±160 degrees; 50 is a clear look without approaching the stop.
_LOOK_BODY_YAW_RAD = {
    "left": math.radians(50),
    "right": math.radians(-50),
    "front": 0.0,
}
_PITCH_DEG = {"up": -30, "down": 30}


class AlignedLookMove(Move):
    """One turn where head yaw stays equal to body yaw at every sample."""

    def __init__(
        self,
        *,
        start_body_yaw: float,
        target_body_yaw: float,
        start_pitch: float,
        target_pitch: float,
        start_antennas: tuple[float, float],
        target_antennas: tuple[float, float],
        duration: float,
        hold_s: float = 0.0,
    ) -> None:
        """Store the yaw, pitch, and antenna endpoints."""
        self.start_body_yaw = start_body_yaw
        self.target_body_yaw = target_body_yaw
        self._start_pitch = start_pitch
        self._target_pitch = target_pitch
        self._start_antennas = start_antennas
        self._target_antennas = target_antennas
        # Extra hold stays on the arrived pose; breathing yaws the base to center when a look ends.
        self._travel_s = duration
        self._duration = duration + (hold_s if hold_s > 0.0 else 0.0)

    @property
    def duration(self) -> float:
        """Duration property required by the Move interface."""
        return self._duration

    def evaluate(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        """Sample one pose whose yaw is the interpolated body yaw."""
        if self._travel_s <= 0:
            progress = 1.0
        else:
            progress = min(1.0, max(0.0, t / self._travel_s))
        # Min-jerk so the base velocity is zero at the start and the stop.
        progress = time_trajectory(progress, method=InterpolationTechnique.MIN_JERK)
        body_yaw = self.start_body_yaw + (self.target_body_yaw - self.start_body_yaw) * progress
        pitch = self._start_pitch + (self._target_pitch - self._start_pitch) * progress
        head_pose = create_head_pose(0, 0, 0, 0, pitch, body_yaw, degrees=False)
        antennas = np.array(
            [
                self._start_antennas[0] + (self._target_antennas[0] - self._start_antennas[0]) * progress,
                self._start_antennas[1] + (self._target_antennas[1] - self._start_antennas[1]) * progress,
            ],
            dtype=np.float64,
        )
        return head_pose, antennas, body_yaw


def _pitch_rad(pose: Any) -> float:
    matrix = np.asarray(pose, dtype=np.float64)
    return float(R.from_matrix(matrix[:3, :3]).as_euler("xyz")[1])


def queue_aligned_look(deps: ToolDependencies, direction: str, hold_s: float = 0.0) -> AlignedLookMove:
    """Queue the aligned look move_head uses and return it."""
    current_head_joints, current_antennas = deps.reachy_mini.get_current_joint_positions()
    current_body_yaw = float(current_head_joints[0])
    if direction in _LOOK_BODY_YAW_RAD or direction in _PITCH_DEG:
        body_yaw = _LOOK_BODY_YAW_RAD.get(direction, current_body_yaw)
        pitch = math.radians(_PITCH_DEG.get(direction, 0))
    else:
        body_yaw = 0.0
        pitch = 0.0
    start_pitch = _pitch_rad(deps.reachy_mini.get_current_head_pose())

    look = AlignedLookMove(
        start_body_yaw=current_body_yaw,
        target_body_yaw=body_yaw,
        start_pitch=start_pitch,
        target_pitch=pitch,
        start_antennas=(float(current_antennas[0]), float(current_antennas[1])),
        target_antennas=(0.0, 0.0),
        duration=deps.motion_duration_s,
        hold_s=hold_s,
    )
    deps.movement_manager.queue_move(look)
    deps.movement_manager.set_moving_state(look.duration)
    return look


class MoveHead(Tool):
    """Look in a direction, turning the base for left and right."""

    name = "move_head"
    DIRECTIONS: tuple[str, ...] = ("left", "right", "up", "down", "front")
    description = (
        "Look left, right, up, down, or front. Left and right rotate the lower body "
        "so the whole robot turns. Up and down tilt only the head and keep the current body angle. "
        "Front returns the head and the body to center."
    )
    needs_response = False
    parameters_schema = {
        "type": "object",
        "properties": {
            "direction": {
                "type": "string",
                "enum": ["left", "right", "up", "down", "front"],
            },
        },
        "required": ["direction"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Look in one move, with head yaw locked to the base."""
        direction_raw = kwargs.get("direction")
        if not isinstance(direction_raw, str):
            return {"error": "direction must be a string"}
        direction = direction_raw
        logger.info("Tool call: move_head direction=%s", direction)

        try:
            queue_aligned_look(deps, direction)
            return {"status": f"looking {direction}"}
        except Exception as e:
            logger.error("move_head failed: %s", e)
            return {"error": f"move_head failed: {type(e).__name__}: {e}"}
