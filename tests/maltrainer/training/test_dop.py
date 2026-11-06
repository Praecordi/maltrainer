"""The DOP caption swap and its log lines, without a trainer."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from maltrainer.training.dop import check_dop_config, dop_captions, dop_triggers

SAMPLES = [("a", "a photo of sks"), ("b", "a landscape")]


def test_the_swap_is_reported_with_an_example_and_the_captions_without_the_trigger_warned_about(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="maltrainer.training.dop"):
        captions = dop_captions(SAMPLES, "sks", "person")

    assert captions == {"a": "a photo of person"}
    info, warning = caplog.records
    assert info.getMessage() == (
        "DOP: 1 of 2 captions name the trigger 'sks'; their DOP caption swaps it for 'person'"
        " (e.g. 'a photo of sks' -> 'a photo of person')"
    )
    assert warning.levelno == logging.WARNING
    assert warning.getMessage().startswith("DOP: 1 caption(s) don't contain the trigger 'sks' and get no DOP pass (")


def test_no_warning_when_every_caption_names_the_trigger(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="maltrainer.training.dop"):
        dop_captions(SAMPLES[:1], "sks", "person")

    assert [record.levelno for record in caplog.records] == [logging.INFO]


def test_several_subjects_each_lose_their_trigger_and_a_caption_naming_both_loses_both(
    caplog: pytest.LogCaptureFixture,
) -> None:
    samples = [
        ("a", "a ayasedi woman, smiling"),
        ("b", "a saisheta woman, nude"),
        ("c", '{"items":[{"caption":"a ayasedi woman"},{"caption":"a saisheta woman"}]}'),
        ("d", "a landscape"),
    ]

    with caplog.at_level(logging.INFO, logger="maltrainer.training.dop"):
        captions = dop_captions(samples, ["ayasedi woman", "saisheta woman"], "woman")

    assert captions == {
        "a": "a woman, smiling",
        "b": "a woman, nude",
        "c": '{"items":[{"caption":"a woman"},{"caption":"a woman"}]}',
    }
    assert (
        caplog.records[0]
        .getMessage()
        .startswith(
            "DOP: 3 of 4 captions name the triggers any of ['ayasedi woman', 'saisheta woman']; their DOP caption"
        )
    )
    assert "don't contain the triggers any of" in caplog.records[1].getMessage()


@pytest.mark.parametrize(
    ("train", "dataset", "expected"),
    [
        ({}, {"trigger": "zed"}, ["zed"]),
        ({}, {"trigger": " "}, []),
        (
            {"dop_triggers": ["ayasedi woman", " saisheta woman "]},
            {"trigger": "zed"},
            ["ayasedi woman", "saisheta woman"],
        ),
        ({"dop_triggers": "ayasedi woman, saisheta woman"}, {}, ["ayasedi woman", "saisheta woman"]),
        ({"dop_triggers": []}, {"trigger": "zed"}, []),
    ],
)
def test_dop_triggers_come_from_train_dop_triggers_else_the_dataset_trigger(
    train: dict[str, Any], dataset: dict[str, Any], expected: list[str]
) -> None:
    assert dop_triggers({"train": train, "dataset": dataset}) == expected


def test_an_empty_dop_triggers_list_is_refused_by_name() -> None:
    config = {"train": {"dop": True, "dop_class": "woman", "dop_triggers": []}, "dataset": {"trigger": "zed"}}

    with pytest.raises(ValueError, match=r"train\.dop_triggers is empty"):
        check_dop_config(config)
