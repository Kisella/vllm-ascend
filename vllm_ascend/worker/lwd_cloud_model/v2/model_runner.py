# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Any

import torch
from vllm.v1.worker.gpu.lwd.model_runner import LwdCloudModelRunnerMixin
from vllm.v1.worker.gpu.model_states.interface import ModelState

from vllm_ascend.models.deepseek_v41.engram.model_state import EngramModelState
from vllm_ascend.worker.lwd_cloud_model.v2.model_state import (
    LwdAscendCloudEngramModelState,
    LwdAscendCloudMambaHybridModelState,
    LwdAscendCloudModelState,
)
from vllm_ascend.worker.lwd_cloud_model.v2.model_wrapper import (
    LwdAscendCloudModelWrapper,
)
from vllm_ascend.worker.lwd_cloud_model.v2.mtp_speculator import (
    LwdAscendCloudMTPSpeculator,
)
from vllm_ascend.worker.v2.model_runner import NPUModelRunner
from vllm_ascend.worker.v2.model_states.default import AscendModelState
from vllm_ascend.worker.v2.model_states.mamba_hybrid import AscendMambaHybridModelState


class LwdAscendCloudModelRunnerV2(LwdCloudModelRunnerMixin, NPUModelRunner):
    def __init__(self, vllm_config: Any, device: torch.device, worker: Any = None) -> None:
        super().__init__(vllm_config, device)
        self.init_lwd_cloud_runner(worker)

    def create_speculator(self) -> Any:
        assert self.speculative_config is not None
        if self.speculative_config.method == "mtp":
            # Native MRV2 attention initialization already handles per-KV-cache-group
            # metadata; the LWD mixin only replaces the prompt-embedding path.
            return LwdAscendCloudMTPSpeculator(self.vllm_config, self.device)
        return super().create_speculator()

    def lwd_model_state_overrides(self) -> dict[type[ModelState], type[ModelState]]:
        return {
            AscendModelState: LwdAscendCloudModelState,
            AscendMambaHybridModelState: LwdAscendCloudMambaHybridModelState,
            EngramModelState: LwdAscendCloudEngramModelState,
        }

    def _install_lwd_model_wrapper(self) -> None:
        self.model = LwdAscendCloudModelWrapper(self.model, self.model_state)

    def get_model(self) -> torch.nn.Module:
        return self.model.get_original_model()

    def _dummy_run(self, *args: Any, **kwargs: Any) -> Any:
        self.model_state.clear_lwd_hash_step()
        return super()._dummy_run(*args, **kwargs)

    def lwd_release_finished(self, finished_req_ids: list[str]) -> None:
        super().lwd_release_finished(finished_req_ids)
        for req_id in finished_req_ids:
            self.model_state.release_lwd_request(req_id)
