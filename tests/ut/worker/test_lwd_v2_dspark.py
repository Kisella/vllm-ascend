# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import torch

from vllm_ascend.worker.lwd_cloud_model.lwd_embed_lookup import LwdEmbedLookup
from vllm_ascend.worker.lwd_cloud_model.lwd_engram_inputs import (
    LwdEngramBatchView,
    LwdEngramDims,
    LwdEngramInputs,
)
from vllm_ascend.worker.lwd_cloud_model.v2.model_runner import LwdAscendCloudModelRunnerV2


def test_runner_uses_native_dspark_speculator(monkeypatch) -> None:
    runner = LwdAscendCloudModelRunnerV2.__new__(LwdAscendCloudModelRunnerV2)
    runner.speculative_config = SimpleNamespace(method="dspark")
    runner.vllm_config = object()
    runner.device = torch.device("cpu")
    selected = object()
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.model_runner.init_speculator",
        lambda vllm_config, device, req_states: selected,
    )

    assert runner.create_speculator() is selected


def test_engram_lookback_uses_device_position_after_draft_rollback() -> None:
    weight = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.bfloat16)
    engram = LwdEngramInputs(
        LwdEngramDims(1, 2, 3, 10, 11),
        torch.device("cpu"),
        LwdEmbedLookup(weight, vocab_start=0, vocab_size=2),
        max_num_tokens=2,
        max_num_reqs=1,
        num_spec_tokens=2,
    )
    engram.add_prompt_chunk("req", 0, 2, [], weight)
    batch = LwdEngramBatchView(
        req_ids=["req"],
        num_computed_tokens=np.array([4], dtype=np.int32),
        num_prompt_tokens=np.array([2], dtype=np.int32),
    )

    window = engram.build_window(
        torch.full((1, 2), -1, dtype=torch.int32),
        batch,
        positions=torch.tensor([2]),
        query_start_loc=torch.tensor([0, 1]),
        stage=lambda values, count: torch.from_numpy(values),
    )

    assert window.tolist() == [[1, 0]]
