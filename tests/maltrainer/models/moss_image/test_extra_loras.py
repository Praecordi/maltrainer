"""Preview-only LoRAs (sample.loras): applied unmerged while a preview renders, gone afterwards."""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from maltrainer.models.moss_image.extra_loras import ExtraLoRAs, specs_from
from maltrainer.models.moss_image.lora import MOSSImageLoRANetwork
from maltrainer.models.moss_image.model import MOSSImageTransformer

Maker = Callable[..., MOSSImageTransformer]
LEAVES = ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")


def _run(model: torch.nn.Module) -> torch.Tensor:
    torch.manual_seed(4)
    return model(torch.randn(1, 128, 2, 2), torch.full((1,), 0.4), torch.randn(1, 3, 8))


def _lora_file(
    tmp_path: Path, model: MOSSImageTransformer, rank: int = 2, alpha: float = 4.0, kohya: bool = True
) -> Path:
    """A ComfyUI-keyed LoRA on every block leaf, with separate q / k / v adapters (unfused sizes)."""
    torch.manual_seed(7)
    down_name, up_name = ("lora_down", "lora_up") if kohya else ("lora_A", "lora_B")
    dim = model.layers[0].proj["o"].in_features
    tensors: dict[str, torch.Tensor] = {}
    for i in range(len(model.layers)):
        for leaf in LEAVES:
            if leaf in ("q", "k", "v"):
                n_in, n_out = dim, dim
            else:
                linear = model.layers[i].proj[leaf]
                n_in, n_out = linear.in_features, linear.out_features
            key = f"diffusion_model.layers.{i}.proj.{leaf}"
            tensors[f"{key}.{down_name}.weight"] = torch.randn(rank, n_in) * 0.1
            tensors[f"{key}.{up_name}.weight"] = torch.randn(n_out, rank) * 0.1
            tensors[f"{key}.alpha"] = torch.tensor(alpha)
    path = tmp_path / ("dmd.safetensors" if kohya else "dmd_ab.safetensors")
    save_file(tensors, str(path))
    return path


def _merged(model: MOSSImageTransformer, path: Path, strength: float, fused: bool) -> MOSSImageTransformer:
    """The reference: the same adapters merged into an fp32 copy of the weights."""
    from safetensors.torch import load_file

    state = load_file(str(path))
    merged = copy.deepcopy(model)
    with torch.no_grad():
        for i, layer in enumerate(merged.layers):
            deltas = {}
            for leaf in LEAVES:
                key = f"diffusion_model.layers.{i}.proj.{leaf}"
                down = state.get(f"{key}.lora_down.weight", state.get(f"{key}.lora_A.weight"))
                up = state.get(f"{key}.lora_up.weight", state.get(f"{key}.lora_B.weight"))
                deltas[leaf] = up @ down * (float(state[f"{key}.alpha"]) / down.shape[0]) * strength
            if fused:
                layer.proj["qkv"].weight += torch.cat([deltas["q"], deltas["k"], deltas["v"]])
                leaves = ("o", "ffn_up_gate", "ffn_down")
            else:
                leaves = LEAVES
            for leaf in leaves:
                layer.proj[leaf].weight += deltas[leaf]
    return merged


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("kohya", [True, False])
def test_inside_applied_the_model_matches_the_adapters_merged_and_outside_it_is_untouched(
    make_model: Maker, tmp_path: Path, fused: bool, kohya: bool
) -> None:
    model = make_model(fused_qkv=fused)
    path = _lora_file(tmp_path, model, kohya=kohya)
    before = _run(model)
    extras = ExtraLoRAs(model, [(str(path), 0.7)], "cpu")

    with extras.applied():
        inside = _run(model)

    torch.testing.assert_close(inside, _run(_merged(model, path, 0.7, fused)), rtol=1e-4, atol=1e-5)
    assert not torch.allclose(inside, before)
    assert torch.equal(_run(model), before)


def test_preview_loras_stack_on_a_training_lora_without_touching_its_weights(make_model: Maker, tmp_path: Path) -> None:
    model = make_model(fused_qkv=True)
    path = _lora_file(tmp_path, model)
    network = MOSSImageLoRANetwork(model, rank=2, targets="all")
    for layer in network.lora_layers.values():
        layer.lora_up.weight.data.normal_(std=0.5)
    trained = {k: v.clone() for k, v in network.get_state_dict(None).items()}
    alone = _run(network)
    extras = ExtraLoRAs(network.transformer, [(str(path), 1.0)], "cpu")

    with extras.applied():
        both = _run(network)

    assert not torch.allclose(both, alone)
    assert torch.equal(_run(network), alone)
    assert all(torch.equal(v, trained[k]) for k, v in network.get_state_dict(None).items())


def test_an_adapter_that_matches_no_module_is_an_error(make_model: Maker, tmp_path: Path) -> None:
    path = tmp_path / "bad.safetensors"
    save_file(
        {
            "diffusion_model.nowhere.lora_down.weight": torch.zeros(1, 4),
            "diffusion_model.nowhere.lora_up.weight": torch.zeros(4, 1),
        },
        str(path),
    )

    with pytest.raises(ValueError, match="match no module"):
        ExtraLoRAs(make_model(), [(str(path), 1.0)], "cpu")


def test_sample_loras_take_paths_or_path_and_strength() -> None:
    assert specs_from(None) == []
    assert specs_from({"loras": ["a.safetensors", {"path": "b.safetensors", "strength": 0.5}]}) == [
        ("a.safetensors", 1.0),
        ("b.safetensors", 0.5),
    ]
