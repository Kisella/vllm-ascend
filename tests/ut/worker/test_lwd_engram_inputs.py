# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu.lwd.cloud_input import (
    LwdCloudBatchView,
    LwdPromptWindow,
    LwdStagedWindow,
)

from vllm_ascend.worker.lwd_cloud_model.lwd_embed_lookup import LwdEmbedLookup
from vllm_ascend.worker.lwd_cloud_model.lwd_engram_inputs import LwdEngramDims, LwdEngramInputs
from vllm_ascend.worker.lwd_cloud_model.lwd_host_stage import LwdReqEmbed
from vllm_ascend.worker.lwd_cloud_model.v2.cloud_input import LwdAscendCloudInputAssembler
from vllm_ascend.worker.lwd_cloud_model.v2.model_state import LwdAscendCloudEngramModelState
from vllm_ascend.worker.lwd_hash.lwd_up_payload import LwdUpPayloadLayout


def test_engram_record_starts_with_a_prefix_hit_chunk() -> None:
    weight = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]], dtype=torch.bfloat16)
    dims = LwdEngramDims(
        num_layers=1,
        num_hash_cols=1,
        max_ngram_size=3,
        image_token_id=10,
        image_pad_token_id=11,
    )
    inputs = LwdEngramInputs(
        dims,
        torch.device("cpu"),
        LwdEmbedLookup(weight, vocab_start=0, vocab_size=4),
        max_num_tokens=4,
        max_num_reqs=1,
        num_spec_tokens=0,
    )

    inputs.add_prompt_chunk("req", start=2, prompt_len=4, spans=[], embeds=weight[2:])

    record = inputs._records["req"]
    assert record.tail_complete
    assert record.tail_cells.tolist() == [3, 2]


def _stage_v2_engram_prompt() -> tuple[LwdAscendCloudInputAssembler, LwdEngramInputs]:
    weight = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.bfloat16)
    hashes = torch.tensor([[[101, 102]], [[103, 104]]], dtype=torch.int32)
    layout = LwdUpPayloadLayout(hidden_size=2, num_engram_layers=1, num_engram_cols=2)
    packed = layout.pack(weight, engram_hashes=hashes)
    dims = LwdEngramDims(
        num_layers=1,
        num_hash_cols=2,
        max_ngram_size=3,
        image_token_id=10,
        image_pad_token_id=11,
    )
    engram = LwdEngramInputs(
        dims,
        torch.device("cpu"),
        LwdEmbedLookup(weight, vocab_start=0, vocab_size=2),
        max_num_tokens=2,
        max_num_reqs=1,
        num_spec_tokens=0,
    )
    assembler = LwdAscendCloudInputAssembler.__new__(LwdAscendCloudInputAssembler)
    super(LwdAscendCloudInputAssembler, assembler).__init__(
        device=torch.device("cpu"),
        hidden_size=2,
        max_num_tokens=2,
        enable_lookahead=False,
    )
    assembler.up_payload_layout = layout
    assembler.lwd_hash_state = None
    assembler._lwd_image_spans = {}
    assembler._lwd_prompt_lens = {}
    assembler.bind_engram_inputs(engram)
    assembler.add_engram_request("req", prompt_len=2, spans=[])
    request = LwdReqEmbed(
        req_id="req",
        host=packed,
        numel=4,
        prompt_rows=2,
        hidden_size=2,
        tail_off=4,
        tail_sizes=(0, 8),
    )
    assembler.prompt_stage = SimpleNamespace(take_req=lambda req_id: request)
    plan = [LwdPromptWindow("req", 0, 0, 2)]
    device_payload = torch.empty_like(packed)
    device_payload[:4].copy_(packed[:4])

    assembler._stage_window_extras(plan, device_payload, 4)

    assert assembler._elements_per_row() == 10
    torch.testing.assert_close(device_payload, packed)
    view = LwdCloudBatchView(
        req_ids=["req"],
        idx_mapping_np=np.array([0]),
        num_scheduled_tokens=np.array([2], dtype=np.int32),
        num_computed_tokens_np=np.array([0], dtype=np.int32),
        prefill_len_np=np.array([2], dtype=np.int32),
        query_start_loc_np=np.array([0, 2], dtype=np.int32),
    )
    assembler._inject_window_extras(LwdStagedWindow(device_payload, None, plan), view)
    assert engram._records["req"].tail_cells.tolist() == [1, 0]
    torch.testing.assert_close(engram._hashes[:2], hashes)
    return assembler, engram


def test_v2_up_engram_segments_stage_and_inject() -> None:
    _stage_v2_engram_prompt()


def test_v2_eager_engram_rebuilds_prefill_and_decode_inputs() -> None:
    _, engram = _stage_v2_engram_prompt()
    state = LwdAscendCloudEngramModelState.__new__(LwdAscendCloudEngramModelState)
    state._cg_mode = CUDAGraphMode.NONE
    state.kvpp_is_dummy_run = False
    state.device = torch.device("cpu")
    state.lwd_engram_inputs = engram
    state.lookback_token_ids = torch.full((1, 2), -1, dtype=torch.int32)
    state._lwd_engram_staging = {}
    state.model = SimpleNamespace(prepare_engram_graph_inputs=lambda num_tokens: {})
    req_states = SimpleNamespace(prompt_len=SimpleNamespace(np=np.array([2], dtype=np.int32)))

    def batch(start: int, token_ids: list[int]) -> SimpleNamespace:
        count = len(token_ids)
        return SimpleNamespace(
            req_ids=["req"],
            num_reqs=1,
            num_tokens=count,
            num_tokens_after_padding=count,
            is_dummy=False,
            num_scheduled_tokens=np.array([count], dtype=np.int32),
            num_computed_tokens_np=np.array([start], dtype=np.int32),
            idx_mapping_np=np.array([0]),
            input_ids=torch.tensor(token_ids, dtype=torch.int32),
            positions=torch.arange(start, start + count, dtype=torch.int64),
            query_start_loc=torch.tensor([0, count], dtype=torch.int32),
        )

    prefill = state.prepare_engram_inputs(batch(0, [99, 99]), req_states)

    assert prefill["input_ids"].tolist() == [0, 0]
    assert engram.device_inputs()["external_rows"][:2].all()
    assert state._engram_inputs["input_ids"].data_ptr() == prefill["input_ids"].data_ptr()
    engram.end_step()

    decode = state.prepare_engram_inputs(batch(2, [7]), req_states)

    assert decode["input_ids"].tolist() == [7]
    assert decode["lookback_token_ids"][0].tolist() == [1, 0]
    assert engram.device_inputs() == {}
    assert state._engram_inputs["input_ids"].data_ptr() == decode["input_ids"].data_ptr()
