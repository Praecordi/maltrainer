"""DOP on a MOSS-Image LoRA run: each subject's caption without its trigger, held to the base model's prediction."""

from __future__ import annotations

import logging
from typing import Any

import pytest
import torch

import train_moss_image_finetune as finetune
import train_moss_image_lora as lora

from .runs import prepared_lora, progress_lines, run_lora
from .world import Workspace, write_image

DOP = {"dop": True, "dop_class": "woman", "dop_triggers": ["ann woman", "bea woman"]}


def _two_subjects(ws: Workspace) -> None:
    ws.add_images("ann", 2, "a ann woman, smiling")
    ws.add_images("bea", 2, "a bea woman, nude")
    write_image(ws.data / "both" / "pair.png", '{"items":[{"caption":"a ann woman"},{"caption":"a bea woman"}]}')


def test_each_subjects_dop_caption_drops_its_trigger_and_is_cached_with_the_training_captions(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    _two_subjects(workspace)
    config = workspace.lora_config(train={**DOP, "dop_every": 2, "dop_multiplier": 2.0})

    with caplog.at_level(logging.INFO):
        trainer = lora.MOSSImageLoRATrainer(config)
        trainer.precache_dataset()

    assert sorted(set(trainer.dop_captions.values())) == [
        "a woman, nude",
        "a woman, smiling",
        '{"items":[{"caption":"a woman"},{"caption":"a woman"}]}',
    ]
    assert set(trainer.dop_captions.values()) <= set(trainer.text_cache)
    assert trainer.run_metadata["dop"] == {
        "triggers": ["ann woman", "bea woman"],
        "class": "woman",
        "multiplier": 2.0,
        "every": 2,
        "captions": 5,
    }
    assert "DOP: 5 of 5 captions name the triggers any of ['ann woman', 'bea woman']" in caplog.text


def test_dop_adds_a_second_loss_that_starts_at_zero_and_leaves_the_adapters_switched_on(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _two_subjects(workspace)
    losses: list[float | None] = []
    original = lora.MOSSImageLoRATrainer._dop_loss

    def spy(self: lora.MOSSImageLoRATrainer, index: int) -> torch.Tensor | None:
        result = original(self, index)
        losses.append(None if result is None else result.item())
        return result

    monkeypatch.setattr(lora.MOSSImageLoRATrainer, "_dop_loss", spy)

    trainer = run_lora(workspace.lora_config(train={**DOP, "max_steps": 3}))

    assert losses[0] == 0.0 and all(loss is not None and loss > 0 for loss in losses[1:]) and len(losses) == 3
    assert trainer.last_dop_loss == losses[-1] and trainer.micro_step == 3
    assert trainer.lora_network._active == [True]
    assert all("DOP: " in line for line in progress_lines(workspace))


def test_dop_every_runs_the_pass_on_every_nth_micro_step_only(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _two_subjects(workspace)
    ran: list[bool] = []
    original = lora.MOSSImageLoRATrainer._dop_loss

    def spy(self: lora.MOSSImageLoRATrainer, index: int) -> torch.Tensor | None:
        result = original(self, index)
        ran.append(result is not None)
        return result

    monkeypatch.setattr(lora.MOSSImageLoRATrainer, "_dop_loss", spy)

    run_lora(workspace.lora_config(train={**DOP, "max_steps": 4, "dop_every": 2}))

    assert ran == [True, False, True, False]


def test_the_dop_loss_is_the_multiplied_distance_to_the_base_models_own_prediction(workspace: Workspace) -> None:
    _two_subjects(workspace)
    trainer = prepared_lora(workspace.lora_config(train={**DOP, "dop_multiplier": 3.0}))
    for parameter in trainer.lora_network.trainable_parameters():
        torch.nn.init.normal_(parameter, std=0.5)
    index = next(i for i, (_, caption) in enumerate(trainer.samples) if caption == "a ann woman, smiling")
    trainer.config["train"]["caption_dropout"] = 0.0
    trainer._training_step(index)
    noisy, sigma = trainer._dop_inputs or (None, None)
    assert noisy is not None and sigma is not None
    context = trainer.text_cache["a woman, smiling"].to(trainer.device, trainer.dtype).unsqueeze(0)
    with torch.no_grad(), trainer.lora_network.disabled():
        base = trainer.lora_network(noisy, sigma, context).float()
    with torch.no_grad():
        adapted = trainer.lora_network(noisy, sigma, context).float()

    loss = trainer._dop_loss(index)

    assert loss is not None and not torch.equal(base, adapted)
    torch.testing.assert_close(loss.detach(), torch.nn.functional.mse_loss(adapted, base) * 3.0)


def test_without_dop_there_is_no_second_pass_and_no_dop_note(workspace: Workspace) -> None:
    _two_subjects(workspace)

    trainer = run_lora(workspace.lora_config(train={"max_steps": 2}))

    assert trainer.dop_captions == {} and trainer.last_dop_loss is None
    assert not any("DOP: " in line for line in progress_lines(workspace))


@pytest.mark.parametrize(
    ("train", "message"),
    [
        ({"dop": True, "dop_class": " "}, r"train\.dop needs train\.dop_class"),
        ({"dop": True, "dop_class": "woman", "dop_triggers": []}, r"train\.dop_triggers is empty"),
    ],
)
def test_a_dop_config_without_class_or_triggers_is_refused(
    workspace: Workspace, train: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        lora.MOSSImageLoRATrainer(workspace.lora_config(train=train))


def test_the_finetune_refuses_dop_and_points_to_prior(workspace: Workspace) -> None:
    with pytest.raises(ValueError, match=r"train\.dop is for MOSS-Image LoRAs"):
        finetune.MOSSImageFinetuneTrainer(workspace.config(train=DOP))
