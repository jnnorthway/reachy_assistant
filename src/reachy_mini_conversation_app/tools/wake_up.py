import asyncio
import logging
from typing import Any

from reachy_mini_conversation_app.sleep_mode import WAKE_POSE_S, hold_sleep, sleep_mode, command_awake_pose
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


def run_wake_motion(deps: ToolDependencies, reason: str) -> dict[str, Any]:
    """Lift to the upright pose and clear sticky sleep when it succeeds."""
    was_quiet = sleep_mode.quiet()
    movement_manager = deps.movement_manager
    try:
        command_awake_pose(deps.reachy_mini, movement_manager, WAKE_POSE_S)
        try:
            deps.reachy_mini.enable_wobbling()
        except Exception as wobble_error:
            logger.debug("Error enabling wobbling after wake: %s", wobble_error)
    except Exception as exc:
        logger.error("wake_up failed: %s", exc)
        if was_quiet:
            try:
                hold_sleep(deps.reachy_mini, movement_manager)
            except Exception as sleep_error:
                logger.error("Could not restore sleep after a failed wake: %s", sleep_error)
        return {"error": f"wake_up failed: {type(exc).__name__}: {exc}"}
    if was_quiet:
        sleep_mode.wake()
    return {"status": "success", "action": "wake_up", "reason": reason}


class WakeUp(Tool):
    """Wake Reachy Mini, including from sticky sleep or jump scare."""

    name = "wake_up"
    description = (
        "Wake Reachy Mini and enable motors. "
        "Use when the user clearly asks Kevin to wake up. This ends sleep and jump scare mode."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Optional short reason for waking Reachy up.",
            }
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Lift Reachy upright and end sticky sleep."""
        reason = kwargs.get("reason", "unspecified")
        if not isinstance(reason, str) or not reason.strip():
            reason = "unspecified"
        logger.info("Tool call: wake_up reason=%s", reason)
        return await asyncio.to_thread(run_wake_motion, deps, reason)
