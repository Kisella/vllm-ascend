# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.lwd.mtp_speculator import LwdCloudMTPSpeculatorMixin

from vllm_ascend.worker.v2.spec_decode.mtp.speculator import AscendMTPSpeculator


class LwdAscendCloudMTPSpeculator(LwdCloudMTPSpeculatorMixin, AscendMTPSpeculator):
    def _prepare_replicated_prefill_attn(
        self, attn_metadata, slot_mappings, num_reqs_padded, num_tokens_padded, cudagraph_runtime_mode
    ):
        batch = self._lwd_draft_batch
        if (
            batch is not None
            and attn_metadata is not None
            and self.pcp_manager is None
            and self.attn_architecture == "DSA"
            and cudagraph_runtime_mode == CUDAGraphMode.NONE
            and batch.num_reqs_after_padding > batch.num_reqs > 0
            and batch.num_tokens < num_tokens_padded
            and not batch.is_prefilling_np[: batch.num_reqs].any()
        ):
            # Target FULL decode may assign tokens to dummy requests. Eager draft
            # attention consumes live Q rows, so build its SAS metadata from the
            # same live request boundaries, using the draft's step-zero buffers.
            attn_metadata = self._build_attn_metadata(
                num_reqs=batch.num_reqs,
                batch_desc=BatchExecutionDescriptor(
                    cg_mode=cudagraph_runtime_mode, num_tokens=batch.num_tokens, num_reqs=batch.num_reqs
                ),
                query_start_loc_np=batch.query_start_loc_np[: batch.num_reqs + 1],
                seq_lens_cpu_upper_bound=batch.seq_lens_cpu_upper_bound,
                step=0,
            )
            return attn_metadata, slot_mappings
        return super()._prepare_replicated_prefill_attn(
            attn_metadata, slot_mappings, num_reqs_padded, num_tokens_padded, cudagraph_runtime_mode
        )
