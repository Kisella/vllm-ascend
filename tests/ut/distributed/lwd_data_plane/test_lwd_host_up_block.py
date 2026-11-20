"""Host UP block sizing follows both DeepSeek V4 Hash layer config layouts."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[4] / "vllm_ascend"
HOST = ROOT / "distributed" / "lwd_data_plane" / "host"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_routing = _load(ROOT / "worker" / "lwd_hash" / "lwd_hash_routing.py", "lwd_hash_routing_ut")
_policy = _load(HOST / "ucx_block_policy.py", "ucx_block_policy_ut")

_tree = ast.parse((HOST / "lwd_host_channels.py").read_text(encoding="utf-8"))
_functions = [
    node
    for node in _tree.body
    if isinstance(node, ast.FunctionDef) and node.name in ("_hf_attr", "_resolve_up_block_mb")
]


def _resolve_up_block(hf_config):
    """Run the real sizing function with the model config stubbed."""
    planned = {}

    def spy(**kwargs):
        planned.update(kwargs)
        return NS(
            warnings=(),
            model_type="",
            max_num_batched_tokens=0,
            per_token_elements=0,
            raw_mib=0.0,
            auto_up_block_mb=0,
            source="",
            up_block_mb=0,
        )

    policy = NS(resolve=spy, LwdUcxBlockPolicyError=_policy.LwdUcxBlockPolicyError)
    namespace = {
        "Any": Any,
        "SimpleNamespace": NS,
        "ucx_block_policy": policy,
        "hash_layer_count": _routing.hash_layer_count,
        "logger": NS(warning=lambda *args, **kwargs: None, info=lambda *args, **kwargs: None),
        "get_current_vllm_config": lambda: NS(
            model_config=NS(hf_config=hf_config),
            scheduler_config=NS(max_num_batched_tokens=100),
        ),
        "_host_plane": lambda: NS(up_buffer_size=None),
    }
    module = ast.fix_missing_locations(ast.Module(body=_functions, type_ignores=[]))
    exec(compile(module, str(HOST / "lwd_host_channels.py"), "exec"), namespace)
    namespace["_resolve_up_block_mb"]()
    return planned


def test_new_layout_feeds_hash_tail_into_up_block():
    planned = _resolve_up_block(
        NS(
            model_type="deepseek_v4",
            hidden_size=4096,
            num_hidden_layers=4,
            mlp_layer_types=["hash_moe", "hash_moe", "moe", "moe"],
            num_experts_per_tok=6,
        )
    )
    assert planned["num_hash_layers"] == 2


def test_legacy_layout_still_feeds_hash_tail():
    planned = _resolve_up_block(
        NS(model_type="deepseek_v4", hidden_size=4096, num_hidden_layers=4, num_hash_layers=2, num_experts_per_tok=6)
    )
    assert planned["num_hash_layers"] == 2


def test_text_config_supplies_layer_layout():
    planned = _resolve_up_block(
        NS(
            model_type="deepseek_v4",
            text_config=NS(
                hidden_size=4096,
                num_hidden_layers=4,
                mlp_layer_types=["hash_moe", "hash_moe", "moe", "moe"],
                num_experts_per_tok=6,
            ),
        )
    )
    assert planned["num_hash_layers"] == 2


def test_non_hash_model_keeps_no_tail():
    planned = _resolve_up_block(NS(model_type="qwen3_5", hidden_size=4096))
    assert planned["num_hash_layers"] == 0


def test_non_prefix_layout_fails_closed():
    with pytest.raises(RuntimeError, match="contiguous prefix"):
        _resolve_up_block(
            NS(
                model_type="deepseek_v4",
                hidden_size=4096,
                num_hidden_layers=4,
                mlp_layer_types=["moe", "hash_moe", "moe", "moe"],
                num_experts_per_tok=6,
            )
        )
