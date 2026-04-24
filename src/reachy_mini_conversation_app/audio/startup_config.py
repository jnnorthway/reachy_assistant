"""Startup configuration for the Reachy Mini audio processor."""

from __future__ import annotations
import logging
from typing import Protocol
from collections.abc import Callable, Sequence

from reachy_mini.media.audio_control_utils import init_respeaker_usb


AudioControlValue = float | int
AudioStartupParameter = tuple[str, tuple[AudioControlValue, ...]]

AUDIO_STARTUP_CONFIG: tuple[AudioStartupParameter, ...] = (
    ("PP_AGCMAXGAIN", (10.0,)),
    ("PP_MIN_NS", (0.8,)),
    ("PP_MIN_NN", (0.8,)),
    ("PP_GAMMA_E", (0.5,)),
    ("PP_GAMMA_ETAIL", (0.5,)),
    ("PP_NLATTENONOFF", (0,)),
    ("PP_MGSCALE", (4.0, 1.0, 1.0)),
)


class ReSpeakerControl(Protocol):
    """Minimal interface used to configure the XVF3800 audio processor."""

    def write(self, name: str, data_list: Sequence[AudioControlValue]) -> None:
        """Write an XVF3800 parameter."""


def apply_audio_startup_config(
    robot: object,
    *,
    logger: logging.Logger | None = None,
    respeaker_factory: Callable[[], ReSpeakerControl | None] = init_respeaker_usb,
) -> bool:
    """Apply the tuned XVF3800 audio configuration for the conversation app."""
    log = logger or logging.getLogger(__name__)
    respeaker = _respeaker_from_robot(robot)
    should_close_respeaker = False

    if respeaker is None:
        log.debug("No existing ReSpeaker control handle on robot media; trying USB discovery.")
        try:
            respeaker = respeaker_factory()
        except Exception as exc:
            log.warning("Skipping Reachy audio startup config: ReSpeaker discovery failed: %s", exc)
            return False
        should_close_respeaker = respeaker is not None

    if respeaker is None:
        log.warning("Skipping Reachy audio startup config: ReSpeaker USB device not found.")
        return False

    failures: list[str] = []
    try:
        for name, values in AUDIO_STARTUP_CONFIG:
            try:
                respeaker.write(name, values)
            except Exception as exc:
                failures.append(f"{name}: {exc}")
                log.warning("Failed to apply audio startup parameter %s=%s: %s", name, _format_values(values), exc)
    finally:
        if should_close_respeaker:
            _close_respeaker(respeaker, log)

    if failures:
        log.warning("Reachy audio startup config completed with %d failed parameter(s).", len(failures))
        return False

    log.info("Applied Reachy audio startup config: %s", _format_config(AUDIO_STARTUP_CONFIG))
    return True


def _respeaker_from_robot(robot: object) -> ReSpeakerControl | None:
    media = getattr(robot, "media", None)
    audio = getattr(media, "audio", None)
    respeaker = getattr(audio, "_respeaker", None)
    return respeaker


def _close_respeaker(respeaker: ReSpeakerControl, logger: logging.Logger) -> None:
    close = getattr(respeaker, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception as exc:
        logger.debug("Error closing temporary ReSpeaker control handle: %s", exc)


def _format_config(config: tuple[AudioStartupParameter, ...]) -> str:
    return ", ".join(f"{name}={_format_values(values)}" for name, values in config)


def _format_values(values: Sequence[AudioControlValue]) -> str:
    return " ".join(str(value) for value in values)
