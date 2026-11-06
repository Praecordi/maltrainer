"""
Extra LoRAs applied to the MOSS-Image DiT for previews only, e.g. an 8-step DMD distillation LoRA, so the
previews render in 8 steps at CFG 1.

They are applied unmerged, through forward hooks that exist only inside ``applied()``: nothing touches the
weights or the LoRA in training, and a distillation delta smaller than bf16's rounding step (lodestones'
MOSS-Image DMD LoRAs) is not lost in a merge. Keys are ComfyUI's, ``diffusion_model.<module>`` with
``lora_down`` / ``lora_up`` (kohya) or ``lora_A`` / ``lora_B`` and an optional ``alpha``. Separate
``proj.q`` / ``proj.k`` / ``proj.v`` adapters on a checkpoint with a fused ``proj.qkv`` are concatenated
onto it, as ComfyUI's loader slices them in.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import nn

logger = logging.getLogger(__name__)

PREFIX = "diffusion_model."
QKV = ("q", "k", "v")


def _pairs(state: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor, float]]:
    """``module -> (down [r, in], up [out, r], alpha / rank)`` for every adapter in a LoRA file."""
    out: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = {}
    for key, down in state.items():
        for down_name, up_name in ((".lora_down.weight", ".lora_up.weight"), (".lora_A.weight", ".lora_B.weight")):
            if key.endswith(down_name):
                base = key[: -len(down_name)]
                up = state[base + up_name]
                alpha = state.get(base + ".alpha")
                rank = down.shape[0]
                scale = float(alpha) / rank if alpha is not None else 1.0
                out[base.removeprefix(PREFIX)] = (down.float(), up.float(), scale)
    return out


class ExtraLoRAs:
    """``specs``: (path, strength) pairs. Weights stay in fp32 on ``device``."""

    def __init__(self, transformer: nn.Module, specs: Sequence[tuple[str, float]], device: torch.device | str):
        modules = dict(transformer.named_modules())
        # module name -> list of (down, up, factor, out_slice or None)
        self.targets: dict[str, list[tuple[torch.Tensor, torch.Tensor, float, slice | None]]] = {}
        for path, strength in specs:
            pairs = _pairs(load_file(str(path)))
            unmatched = []
            for name, (cpu_down, cpu_up, scale) in pairs.items():
                down, up = cpu_down.to(device), cpu_up.to(device)
                factor = float(strength) * scale
                leaf = name.rsplit(".", 1)[-1]
                fused = f"{name.rsplit('.', 1)[0]}.qkv"
                if name in modules:
                    self.targets.setdefault(name, []).append((down, up, factor, None))
                elif leaf in QKV and fused in modules:
                    third = int(modules[fused].out_features) // 3
                    part = QKV.index(leaf)
                    self.targets.setdefault(fused, []).append(
                        (down, up, factor, slice(part * third, (part + 1) * third))
                    )
                else:
                    unmatched.append(name)
            if unmatched:
                raise ValueError(f"{path}: {len(unmatched)} adapters match no module, e.g. {unmatched[:3]}")
            logger.info("[sample] preview LoRA %s at %s: %d adapters", Path(path).name, strength, len(pairs))
        self.modules = {name: modules[name] for name in self.targets}

    @staticmethod
    def _hook(adapters: list[tuple[torch.Tensor, torch.Tensor, float, slice | None]]) -> Any:
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
            x = inputs[0].float()
            delta = torch.zeros(output.shape, dtype=torch.float32, device=output.device)
            for down, up, factor, part in adapters:
                d = (x @ down.T) @ up.T * factor
                if part is None:
                    delta += d
                else:
                    delta[..., part] += d
            return (output.float() + delta).to(output.dtype)

        return hook

    @contextlib.contextmanager
    def applied(self) -> Iterator[None]:
        handles = [
            module.register_forward_hook(self._hook(self.targets[name])) for name, module in self.modules.items()
        ]
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()


def specs_from(sample_config: dict[str, Any] | None) -> list[tuple[str, float]]:
    """``sample.loras``: a list of paths or ``{path, strength}`` (strength defaults to 1.0)."""
    out = []
    for item in (sample_config or {}).get("loras") or []:
        if isinstance(item, dict):
            out.append((str(item["path"]), float(item.get("strength", 1.0))))
        else:
            out.append((str(item), 1.0))
    return out
