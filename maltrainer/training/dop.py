"""Differential output preservation (ai-toolkit's diff_output_preservation).

On the same noised latent and sigma as the training pass, the sample's caption with the trigger replaced by
the class word: the base model's prediction for it (adapters off, no grad) is the target for the LoRA's.

Several subjects in one LoRA (``train.dop_triggers``) each have their trigger replaced by the class word, and a
caption naming two of them (a photo of both) has both replaced, as Fizgig's ``--dop_trigger a, b`` does.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import torch

logger = logging.getLogger(__name__)


def dop_triggers(config: dict[str, Any]) -> list[str]:
    """The words DOP removes: ``train.dop_triggers`` (a list, or comma-separated), else ``dataset.trigger``."""
    listed = config.get("train", {}).get("dop_triggers")
    if listed is None:
        trigger = str(config.get("dataset", {}).get("trigger", "") or "").strip()
        return [trigger] if trigger else []
    items = listed.split(",") if isinstance(listed, str) else listed
    return [str(item).strip() for item in items if str(item).strip()]


def check_dop_config(config: dict[str, Any]) -> None:
    """train.dop needs the class word and the trigger(s) it replaces."""
    train_config = config.get("train", {})
    if not train_config.get("dop"):
        return
    if not str(train_config.get("dop_class", "") or "").strip():
        raise ValueError("train.dop needs train.dop_class, the class word that replaces the trigger")
    if not dop_triggers(config):
        if train_config.get("dop_triggers") is not None:
            raise ValueError("train.dop_triggers is empty: list the trigger of every subject")
        raise ValueError("train.dop needs dataset.trigger: DOP's captions are the captions without it")


def dop_captions(samples: list[tuple[str, str]], trigger: str | Sequence[str], cls: str) -> dict[str, str]:
    """Sample key -> its caption with every trigger in ``trigger`` (one, or several subjects') replaced by ``cls``,
    for the captions that name at least one of them."""
    triggers = [trigger] if isinstance(trigger, str) else list(trigger)
    named = repr(triggers[0]) if len(triggers) == 1 else f"any of {triggers!r}"
    what = "the trigger" if len(triggers) == 1 else "the triggers"

    def swapped(caption: str) -> str:
        for word in triggers:
            caption = caption.replace(word, cls)
        return caption

    captions = {key: swapped(caption) for key, caption in samples if any(word in caption for word in triggers)}
    example = next(((c, captions[k]) for k, c in samples if k in captions), None)
    logger.info(
        "DOP: %d of %d captions name %s %s; their DOP caption swaps it for %r%s",
        len(captions),
        len(samples),
        what,
        named,
        cls,
        f" (e.g. {example[0]!r} -> {example[1]!r})" if example else "",
    )
    missing = len(samples) - len(captions)
    if missing:
        logger.warning(
            "DOP: %d caption(s) don't contain %s %s and get no DOP pass (holding them to the "
            "base model would unlearn the subject).",
            missing,
            what,
            named,
        )
    if not captions:
        raise ValueError(f"train.dop: no caption contains {what} {named}")
    return captions


def dop_loss(network: Any, adapters: Any, inputs: tuple[torch.Tensor, ...], multiplier: float) -> torch.Tensor:
    """The MSE between the LoRA's prediction for ``inputs`` and the base model's (``adapters.disabled()``)."""
    with torch.no_grad(), adapters.disabled():
        target = network(*inputs).float()
    pred = network(*inputs)
    return torch.nn.functional.mse_loss(pred.float(), target) * multiplier
