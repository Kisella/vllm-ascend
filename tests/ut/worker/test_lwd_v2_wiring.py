# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import numpy as np
import torch
from vllm.config.lwd import LwdConnectionId
from vllm.v1.worker.gpu.lwd.cloud_input import LwdCloudBatchView

from vllm_ascend.worker.lwd_cloud_model.v2.cloud_input import (
    LwdAscendCloudInputAssembler,
)


def test_non_endpoint_receives_payload_and_lookahead(monkeypatch) -> None:
    assembler = LwdAscendCloudInputAssembler.__new__(LwdAscendCloudInputAssembler)
    super(LwdAscendCloudInputAssembler, assembler).__init__(
        device=torch.device("cpu"),
        hidden_size=2,
        max_num_tokens=4,
        enable_lookahead=True,
    )
    assembler.lwd_hash_state = None
    assembler.num_hash_layers = 0
    assembler.num_experts_per_tok = 0
    view = LwdCloudBatchView(
        req_ids=["r1"],
        idx_mapping_np=np.array([3]),
        num_scheduled_tokens=np.array([2], dtype=np.int32),
        num_computed_tokens_np=np.array([0], dtype=np.int32),
        prefill_len_np=np.array([4], dtype=np.int32),
        query_start_loc_np=np.array([0, 2], dtype=np.int32),
    )
    broadcasts = []

    def broadcast(tensor, *, src, group, async_op):
        broadcasts.append((tuple(tensor.shape), src, group, async_op))
        if tensor.ndim == 1:
            tensor.copy_(torch.arange(1, 5, dtype=torch.bfloat16))
        else:
            tensor.fill_(8)
        return SimpleNamespace(wait=lambda: None)

    group = SimpleNamespace(world_size=2, ranks=[4, 5], device_group="hccl")
    monkeypatch.setattr(
        "vllm_ascend.worker.lwd_cloud_model.v2.cloud_input.get_tp_group",
        lambda: group,
    )
    monkeypatch.setattr("torch.distributed.broadcast", broadcast)
    monkeypatch.setattr("torch.cuda.current_stream", lambda: SimpleNamespace())
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, stream: None)
    inputs_embeds = torch.zeros((2, 2), dtype=torch.bfloat16)
    next_prompt_embeds = torch.zeros((4, 2), dtype=torch.bfloat16)

    assembler.stage_prompt_window(view, inputs_embeds, next_prompt_embeds)

    assert inputs_embeds.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert next_prompt_embeds[3].tolist() == [8.0, 8.0]
    assert broadcasts == [((4,), 4, "hccl", True), ((1, 2), 4, "hccl", True)]


class _LocalPromptStage:
    def __init__(self) -> None:
        self.host = torch.arange(1, 9, dtype=torch.bfloat16)
        self.consumed = 0

    def take_req(self, req_id: str) -> SimpleNamespace:
        assert req_id == "e0-r1"
        return SimpleNamespace(host=self.host)

    def recv_rows(self, req_id: str) -> int:
        return 4

    def consumed_rows(self, req_id: str) -> int:
        return self.consumed

    def segment_for(self, req_id: str, n_rows: int) -> tuple[int, int, int]:
        return self.consumed * 2, n_rows * 2, 0

    def consume(self, req_id: str, n_rows: int) -> None:
        self.consumed += n_rows


def test_mixed_connections_use_their_own_payload_and_lookahead_sources(monkeypatch) -> None:
    assembler = LwdAscendCloudInputAssembler.__new__(LwdAscendCloudInputAssembler)
    super(LwdAscendCloudInputAssembler, assembler).__init__(
        device=torch.device("cpu"),
        hidden_size=2,
        max_num_tokens=4,
        enable_lookahead=True,
    )
    assembler.lwd_hash_state = None
    assembler.num_hash_layers = 0
    assembler.num_experts_per_tok = 0
    connections = [LwdConnectionId(0, 0, 0), LwdConnectionId(1, 0, 0)]
    stage = _LocalPromptStage()
    assembler.bind_prompt_stage(
        stage,
        lambda req_id: connections[int(req_id[1])],
        {connections[0]: 4, connections[1]: 5},
        local_rank=4,
    )
    view = LwdCloudBatchView(
        req_ids=["e0-r1", "e1-r2"],
        idx_mapping_np=np.array([0, 1]),
        num_scheduled_tokens=np.array([2, 2], dtype=np.int32),
        num_computed_tokens_np=np.array([0, 0], dtype=np.int32),
        prefill_len_np=np.array([4, 4], dtype=np.int32),
        query_start_loc_np=np.array([0, 2, 4], dtype=np.int32),
    )
    broadcasts = []

    def broadcast(tensor, *, src, group, async_op):
        if tensor.ndim == 1 and src == 5:
            tensor.copy_(torch.arange(10, 14, dtype=torch.bfloat16))
        elif tensor.ndim == 2:
            tensor.fill_(8 if src == 4 else 9)
        broadcasts.append((tensor.clone(), src))
        return SimpleNamespace(wait=lambda: None)

    class _Stream:
        def wait_event(self, event) -> None:
            pass

    class _Event:
        def record(self, stream=None) -> None:
            pass

    monkeypatch.setattr("torch.distributed.broadcast", broadcast)
    monkeypatch.setattr(
        "vllm_ascend.worker.lwd_cloud_model.v2.cloud_input.get_tp_group",
        lambda: SimpleNamespace(world_size=2, ranks=[4, 5], device_group="hccl"),
    )
    monkeypatch.setattr("torch.cuda.Stream", lambda device: _Stream())
    monkeypatch.setattr("torch.cuda.Event", _Event)
    monkeypatch.setattr("torch.cuda.stream", lambda stream: __import__("contextlib").nullcontext())
    monkeypatch.setattr("torch.cuda.current_stream", lambda: _Stream())
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, stream: None)
    inputs_embeds = torch.zeros((4, 2), dtype=torch.bfloat16)
    next_prompt_embeds = torch.zeros((2, 2), dtype=torch.bfloat16)

    assembler.stage_prompt_window(view, inputs_embeds, next_prompt_embeds)

    assert inputs_embeds.tolist() == [[1.0, 2.0], [3.0, 4.0], [10.0, 11.0], [12.0, 13.0]]
    assert next_prompt_embeds.tolist() == [[8.0, 8.0], [9.0, 9.0]]
    assert [src for _tensor, src in broadcasts] == [4, 4, 5, 5]
    assert broadcasts[0][0].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert broadcasts[2][0].tolist() == [10.0, 11.0, 12.0, 13.0]
    assert stage.consumed == 2
