# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Any

import numpy as np
import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.gpu.lwd.model_state import LwdCloudModelStateMixin
from vllm.v1.worker.gpu.lwd.pinned_staging import LwdPinnedStaging
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.states import RequestState

from vllm_ascend.models.deepseek_v41.engram.model_state import EngramModelState
from vllm_ascend.worker.lwd_cloud_model.lwd_engram_inputs import (
    LwdEngramBatchView,
    LwdEngramDims,
    LwdEngramInputs,
    LwdStageToDevice,
    lwd_embed_lookup,
)
from vllm_ascend.worker.lwd_cloud_model.v2.cloud_input import (
    LwdAscendCloudInputAssembler,
)
from vllm_ascend.worker.v2.attn_utils import ring_state_update_skipped
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.model_states.default import AscendModelState
from vllm_ascend.worker.v2.model_states.mamba_hybrid import AscendMambaHybridModelState


class LwdAscendCloudModelState(LwdCloudModelStateMixin, AscendModelState):
    def __init__(
        self,
        vllm_config: VllmConfig,
        model: torch.nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        enable_lookahead = vllm_config.num_prefill_lookahead_tokens > 0
        self.init_lwd_cloud_state(enable_lookahead)
        self.lwd_input_assembler = LwdAscendCloudInputAssembler(
            vllm_config=vllm_config,
            device=device,
            hidden_size=self.inputs_embeds_size,
            max_num_tokens=self.max_num_tokens,
            enable_lookahead=enable_lookahead,
        )

    def prepare_inputs(self, input_batch: AscendInputBatch, req_states: RequestState) -> dict[str, Any]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        model_inputs.update(self.lwd_input_assembler.hash_model_inputs(input_batch.num_tokens_after_padding))
        return model_inputs

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        model_inputs.update(self.lwd_input_assembler.hash_model_inputs(num_tokens))
        return model_inputs

    def clear_lwd_hash_step(self) -> None:
        self.lwd_input_assembler.clear_hash_step()

    def release_lwd_request(self, req_id: str) -> None:
        self.lwd_input_assembler.release_hash_request(req_id)

    def publish_lwd_hash_forward_context(self, context: Any, num_tokens: int) -> None:
        self.lwd_input_assembler.publish_hash_forward_context(context, num_tokens)


class LwdAscendCloudMambaHybridModelState(LwdAscendCloudModelState, AscendMambaHybridModelState):
    """LWD prompt injection and sampling with native hybrid-state execution."""


class LwdAscendCloudEngramModelState(LwdAscendCloudModelState, EngramModelState):
    def __init__(
        self,
        vllm_config: VllmConfig,
        model: torch.nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        dims = LwdEngramDims.from_model_config(self.model_config)
        if dims is None:
            raise ValueError("[LWD] Engram state requires a model with Engram layers")
        self.lwd_engram_inputs = LwdEngramInputs(
            dims,
            device,
            lwd_embed_lookup(model),
            self.max_num_tokens,
            self.max_num_reqs,
            vllm_config.num_speculative_tokens,
        )
        self.lwd_input_assembler.bind_engram_inputs(self.lwd_engram_inputs)
        self._lwd_engram_staging: dict[str, LwdPinnedStaging] = {}

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        self.lwd_input_assembler.add_engram_request(
            new_req_data.req_id,
            new_req_data.prompt_len,
            list(new_req_data.lwd_image_placeholder_spans),
        )

    def release_lwd_request(self, req_id: str) -> None:
        super().release_lwd_request(req_id)
        self.lwd_input_assembler.release_engram_request(req_id)

    def clear_lwd_hash_step(self) -> None:
        super().clear_lwd_hash_step()
        self.lwd_engram_inputs.end_step()

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        self.lwd_engram_inputs.end_step()
        return super().prepare_dummy_inputs(num_reqs, num_tokens)

    def finish_execution(self, *, failed: bool) -> None:
        try:
            super().finish_execution(failed=failed)
        finally:
            self.lwd_engram_inputs.end_step()

    def _lwd_engram_staging_for(self, name: str, dtype: torch.dtype, capacity: int) -> LwdStageToDevice:
        staging = self._lwd_engram_staging.get(name)
        if staging is None:
            staging = LwdPinnedStaging(dtype=dtype, min_capacity=capacity, device=self.device)
            self._lwd_engram_staging[name] = staging
        return staging.stage_to_device

    def _lwd_engram_batch_view(self, input_batch: AscendInputBatch, req_states: RequestState) -> LwdEngramBatchView:
        num_reqs = input_batch.num_reqs
        req_indices = input_batch.idx_mapping_np[:num_reqs]
        return LwdEngramBatchView(
            req_ids=input_batch.req_ids[:num_reqs],
            num_computed_tokens=input_batch.num_computed_tokens_np[:num_reqs],
            num_prompt_tokens=np.asarray(req_states.prompt_len.np[req_indices]),
        )

    def prepare_engram_inputs(self, input_batch: AscendInputBatch, req_states: RequestState) -> dict[str, Any]:
        if self._cg_mode not in (None, CUDAGraphMode.NONE):
            raise ValueError("[LWD] MRV2 DeepSeek V4.1 Engram currently requires cudagraph_mode=NONE")
        engram = self.lwd_engram_inputs
        num_tokens = input_batch.num_tokens
        padded_tokens = input_batch.num_tokens_after_padding
        dummy = input_batch.is_dummy or self.kvpp_is_dummy_run or ring_state_update_skipped()
        if dummy:
            engram.end_step()
        else:
            engram.plan_step(
                self._lwd_engram_batch_view(input_batch, req_states),
                input_batch.num_scheduled_tokens,
                self._lwd_engram_staging_for("engram_row_codes", torch.int32, engram.row_codes_capacity),
            )
        input_ids = engram.model_input_ids(input_batch.input_ids, padded_tokens)
        model_inputs: dict[str, Any] = {"input_ids": input_ids}
        window = self.lookback_token_ids
        if window is None:
            return model_inputs
        if dummy:
            window.fill_(-1)
        else:
            engram.build_window(
                window,
                self._lwd_engram_batch_view(input_batch, req_states),
                input_batch.positions,
                input_batch.query_start_loc,
                self._lwd_engram_staging_for("engram_window", torch.int64, engram.window_capacity),
            )
        model_inputs["lookback_token_ids"] = window
        self._engram_inputs = {
            "input_ids": input_ids[:num_tokens],
            "positions": input_batch.positions[:num_tokens],
            "padded_tokens": padded_tokens,
            "lookback_token_ids": window,
            "query_start_loc": None if dummy else input_batch.query_start_loc[: input_batch.num_reqs + 1],
            "cg_mode": self._cg_mode,
            "force_dummy": dummy,
            "token_indices": None,
            **engram.device_inputs(),
        }
        model_inputs.update(self.model.prepare_engram_graph_inputs(padded_tokens))
        return model_inputs
