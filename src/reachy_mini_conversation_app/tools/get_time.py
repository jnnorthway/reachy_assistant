import logging
import asyncio
import datetime
import zoneinfo
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.tools.get_location import _fetch_location, _GEOLOCATION_TIMEOUT_S


logger = logging.getLogger(__name__)


class GetTime(Tool):
    """Return the current date and time in the local timezone based on IP location."""

    name = "get_time"
    description = "Get the current date and time, using the device's location to determine the correct timezone."
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        # Try to resolve timezone from IP location
        tz = None
        try:
            location = await asyncio.wait_for(
                asyncio.to_thread(_fetch_location),
                timeout=_GEOLOCATION_TIMEOUT_S + 1,
            )
            tz_name = location.get("timezone")
            if tz_name:
                tz = zoneinfo.ZoneInfo(tz_name)
        except Exception as e:
            logger.warning("get_time: could not resolve location timezone, falling back to system: %s", e)

        now = datetime.datetime.now(tz=tz).astimezone(tz) if tz else datetime.datetime.now().astimezone()
        return {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M"),
            "timezone": now.strftime("%Z"),
            "day_of_week": now.strftime("%A"),
        }
