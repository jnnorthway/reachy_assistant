"""Phrase detection for sticky sleep and jump scare."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.sleep_mode import (
    SCARE_LINE_ONE,
    SCARE_LINE_TWO,
    SCARE_INSTRUCTIONS,
    SleepMode,
    hold_sleep,
    sleep_mode,
    exact_scare_response,
    quiet_transcript_action,
)


@pytest.mark.parametrize(
    ("transcript", "asleep", "jump_scare", "scare_running", "expected"),
    [
        ("elephant", True, True, False, "scare"),
        ("ELEPHANT!", True, True, False, "scare"),
        ("I saw an elephant", True, True, False, "scare"),
        ("kevin wake up", True, False, False, "wake"),
        ("Wake up", True, True, False, "wake"),
        ("please wake up now", True, False, False, "wake"),
        ("kevin, wake up!", True, True, False, "wake"),
        ("wake up the elephant", True, True, False, "wake"),
        ("wake up", True, True, True, "defer_wake"),
    ],
)
def test_elephant_and_wake_phrases(
    transcript: str,
    asleep: bool,
    jump_scare: bool,
    scare_running: bool,
    expected: str,
) -> None:
    """Elephant is a whole word, and a clear wake phrase leaves sleep."""
    assert (
        quiet_transcript_action(
            transcript,
            asleep=asleep,
            jump_scare=jump_scare,
            scare_running=scare_running,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("transcript", "asleep", "jump_scare", "scare_running"),
    [
        ("elephantine", True, True, False),
        ("elephants", True, True, False),
        ("hello", True, True, False),
        ("what's up", True, False, False),
        ("look around", True, True, False),
        ("sleep", True, False, False),
        ("kevin", True, True, False),
        ("elephant", True, False, False),
        ("wake up", False, False, False),
        ("elephant", False, False, False),
        ("elephant", True, True, True),
        ("nice weather", True, True, True),
    ],
)
def test_other_words_do_not_wake_or_retrigger(
    transcript: str,
    asleep: bool,
    jump_scare: bool,
    scare_running: bool,
) -> None:
    """Unrelated speech, partial words, and a second elephant do not wake Kevin."""
    action = quiet_transcript_action(
        transcript,
        asleep=asleep,
        jump_scare=jump_scare,
        scare_running=scare_running,
    )
    assert action in {"ignore", "respond"}
    if asleep or jump_scare or scare_running:
        assert action != "wake"
        assert action != "scare"


def test_scare_script_is_exact() -> None:
    """The forced reply is the two scare sentences and nothing else."""
    assert SCARE_LINE_ONE == "BOO!"
    assert SCARE_LINE_TWO == "Ha gotcha!"
    assert "BOO! Ha gotcha!" in SCARE_INSTRUCTIONS
    response = exact_scare_response()
    assert response["conversation"] == "none"
    assert response["tool_choice"] == "none"
    assert response["input"][0]["content"][0]["text"] == "BOO! Ha gotcha!"


def test_jump_scare_does_not_retrigger() -> None:
    """A second elephant is ignored until the current scare finishes."""
    mode = SleepMode()
    mode.enter_jump_scare()
    assert mode.begin_scare()
    assert not mode.begin_scare()
    mode.end_scare()
    assert mode.in_jump_scare()
    assert mode.begin_scare()


def test_hold_sleep_stays_in_the_running_app() -> None:
    """Sleep moves to the pose and keeps listening instead of stopping the app."""
    sleep_mode.reset()
    sleep_mode.set_notifier(None)
    robot = MagicMock()
    movement = MagicMock()
    try:
        result = hold_sleep(robot, movement)
        assert result == {"status": "sleeping", "listening": True}
        assert sleep_mode.quiet()
        assert not sleep_mode.in_jump_scare()
    finally:
        sleep_mode.reset()
        sleep_mode.set_notifier(None)
    movement.stop.assert_called_once_with(reset_to_neutral=False)
    robot.goto_sleep.assert_called_once_with()
    robot.disable_wobbling.assert_called_once_with()


def test_sleep_state_reloads(tmp_path: Path) -> None:
    """Sticky sleep and jump scare survive a process restart."""
    path = tmp_path / "sleep_state.json"
    saved = SleepMode()
    saved.configure(path)
    saved.enter_jump_scare()
    loaded = SleepMode()
    loaded.configure(path)
    loaded.load()
    assert loaded.quiet()
    assert loaded.in_jump_scare()
    loaded.wake()
    saved.load()
    assert not saved.quiet()
    assert not saved.in_jump_scare()
