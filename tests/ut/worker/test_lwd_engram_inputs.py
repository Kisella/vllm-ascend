# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_ascend.worker.lwd_cloud_model.lwd_embed_lookup import LwdEmbedLookup
from vllm_ascend.worker.lwd_cloud_model.lwd_engram_inputs import LwdEngramDims, LwdEngramInputs


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
