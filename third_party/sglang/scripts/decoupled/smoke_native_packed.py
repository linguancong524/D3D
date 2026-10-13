#!/usr/bin/env python3
"""H20 smoke test for the native D3DV2 packed Scheduler operation."""

from sglang.srt.speculative.decoupled.engine_backend import DllmEngineBackend
from sglang.srt.speculative.decoupled.protocol import Commit, proposal_identity


def main() -> None:
    backend = DllmEngineBackend(
        model_path="/workspace/models/d3d-1.7b-b16_for_qwen3-8b",
        tokenizer_path="/workspace/models/Qwen3-8B",
        block_size=16,
        num_anchors=2,
        max_branches=6,
        accept_counts=(1, 2, 3),
        dllm_algorithm="OneStep",
        dllm_algorithm_config="/tmp/sdar_dllm.yaml",
        context_length=8192,
        max_running_requests=192,
        chunked_prefill_size=1024,
        attention_backend="flashinfer",
        dtype="bfloat16",
        mem_fraction_static=0.65,
        enable_cuda_graph=False,
        trust_remote_code=True,
    )
    try:
        prefix = (151643, 8948, 198, 151644)
        proposal = backend.draft(session="smoke", revision=0, prefix=prefix)
        outcomes = backend.speculate(
            session="smoke", revision=0, prefix=prefix, proposal=proposal
        )
        print("proposal_len", len(proposal.tokens))
        print("proposal_head", proposal.tokens[:4])
        print("top_rows", len(proposal.top_tokens))
        print("outcome_count", len(outcomes))
        print(
            "branches",
            sorted((b.num_accept_tokens, b.bonus_token) for b in outcomes),
        )
        print("outcome_lens", sorted(len(p.tokens) for p in outcomes.values()))
        anchored = next(branch for branch in outcomes if branch.bonus_token is not None)
        committed = (
            prefix[:-1]
            + proposal.tokens[: anchored.num_accept_tokens]
            + (anchored.bonus_token,)
        )
        backend.commit_many(
            items=[
                Commit(
                    session="smoke",
                    revision=1,
                    proposal_id=proposal_identity(proposal),
                    committed_prefix=committed,
                )
            ]
        )
        second = backend.speculate(
            session="smoke",
            revision=1,
            prefix=committed,
            proposal=outcomes[anchored],
        )
        print("second_outcome_count", len(second))
        print("physical_forward_count", backend.packed_physical_forward_count)
        if len(proposal.tokens) != 16:
            raise RuntimeError("required draft did not return one complete block")
        if len(outcomes) != 6:
            raise RuntimeError("packed replay did not return every selected branch")
        if len(second) != 6:
            raise RuntimeError("retained-KV replay did not return every branch")
        if backend.packed_physical_forward_count != 2:
            raise RuntimeError("each packed replay must use one physical forward")
        backend.close(session="smoke")
    finally:
        backend.shutdown()


if __name__ == "__main__":
    main()
