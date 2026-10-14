import argparse
import json
import tempfile
import time
import uuid
from pathlib import Path

import zmq
from sglang.srt.managers.io_struct import (
    DllmPackedBranch,
    DllmPackedItem,
    DllmPackedReqInput,
)
from sglang.srt.speculative.decoupled.engine_backend import DllmEngineBackend
from sglang.srt.speculative.decoupled.protocol import (
    Commit,
    proposal_identity,
    select_outcome,
)


class StrictDraftService:
    def __init__(self, backend, *, speculate=True):
        self.backend = backend
        self.speculate_enabled = speculate
        self.sessions = {}
        self.metrics = {
            "required_forward_count": 0,
            "required_seconds": 0.0,
            "remote_hit_count": 0,
            "remote_miss_count": 0,
            "optional_forward_count": 0,
            "optional_seconds": 0.0,
        }

    def required(self, prefix):
        if len(prefix) < 2:
            raise ValueError("native remote replay requires at least two prefix tokens")
        session = "required-" + uuid.uuid4().hex
        started = time.perf_counter()
        try:
            response = self.backend.engine.dllm_packed_replay(
                DllmPackedReqInput(
                    rid=session,
                    items=[DllmPackedItem(
                        session=session,
                        revision=0,
                        proposal_id=session,
                        committed_tokens=list(prefix[:-2]),
                        proposal_tokens=[prefix[-2]],
                        branches=[DllmPackedBranch(
                            num_accept_tokens=1, bonus_token=prefix[-1]
                        )],
                    )],
                    branch_token_count=self.backend.block_size,
                    topk=max(2, self.backend.candidate_topk),
                )
            )
            if not response.success:
                raise RuntimeError(response.error)
            from sglang.srt.speculative.decoupled.protocol import Proposal

            outcome = response.items[0].outcomes[0]
            return Proposal(
                tuple(outcome.tokens),
                tuple(tuple(row) for row in outcome.top_tokens),
            )
        finally:
            self.backend.close(session=session)
            self.metrics["required_forward_count"] += 1
            self.metrics["required_seconds"] += time.perf_counter() - started

    def step(self, items):
        live = {item["rid"] for item in items}
        for rid in self.sessions.keys() - live:
            self.backend.close(session=self.sessions.pop(rid)["session"])
        proposals = {}
        pending = []
        for item in items:
            rid = item["rid"]
            prompt, outputs = tuple(item["prompt"]), tuple(item["outputs"])
            if not outputs:
                raise ValueError("missing Target anchor")
            prefix = prompt + outputs
            previous = self.sessions.get(rid)
            proposal = None
            revision = 0
            session = uuid.uuid4().hex
            if previous is not None:
                if (
                    prompt != previous["prompt"]
                    or outputs[:len(previous["outputs"])] != previous["outputs"]
                ):
                    self.backend.close(session=previous["session"])
                else:
                    session = previous["session"]
                    revision = previous["revision"] + 1
                    proposal = select_outcome(
                        previous=previous["proposal"],
                        delta=outputs[len(previous["outputs"]):],
                        outcomes=previous["outcomes"],
                    )
                    if self.speculate_enabled:
                        self.backend.commit_many(items=[Commit(
                            session=session,
                            revision=revision,
                            proposal_id=proposal_identity(previous["proposal"]),
                            committed_prefix=prefix,
                        )])
            if proposal is None:
                self.metrics["remote_miss_count"] += 1
                proposal = self.required(prefix)
            else:
                self.metrics["remote_hit_count"] += 1
            self.sessions[rid] = {
                "session": session, "revision": revision,
                "prompt": prompt, "outputs": outputs,
                "proposal": proposal, "outcomes": {},
            }
            proposals[rid] = {"tokens": list(proposal.tokens)}
            pending.append((session, revision, prefix, proposal))
        return {"proposals": proposals}, pending

    def speculate(self, pending):
        if not self.speculate_enabled:
            return
        started = time.perf_counter()
        outcomes = self.backend.speculate_many(items=pending)
        for state in self.sessions.values():
            if state["session"] in outcomes:
                state["outcomes"] = outcomes[state["session"]]
        self.metrics["optional_forward_count"] += 1
        self.metrics["optional_seconds"] += time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:17891")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--anchors", type=int, default=1)
    parser.add_argument("--max-branches", type=int, default=48)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="flashinfer")
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--mem-fraction", type=float, default=0.8)
    parser.add_argument("--no-speculate", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--packed-graph-max-tokens", type=int)
    parser.add_argument("--packed-graph-max-requests", type=int, default=1)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--metrics-file")
    args = parser.parse_args()
    if args.metrics_file:
        Path(args.metrics_file).parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="d3d-remote-config-") as directory:
        config_path = Path(directory) / "algorithm.json"
        config_path.write_text(json.dumps({"block_size": args.block_size}))
        backend = DllmEngineBackend(
            model_path=args.model, tokenizer_path=args.tokenizer,
            block_size=args.block_size, num_anchors=args.anchors,
            max_branches=args.max_branches, candidate_topk=args.anchors + 1,
            accept_counts=tuple(range(1, args.block_size + 1)),
            context_length=args.context_length,
            max_running_requests=max(4, args.packed_graph_max_requests + 1),
            dtype=args.dtype, attention_backend=args.attention_backend,
            trust_remote_code=True, mem_fraction_static=args.mem_fraction,
            enable_cuda_graph=args.cuda_graph, packed_single_row=True,
            batched_branch_topk=True, packed_causal_pending=True,
            branch_group_size=max(1, args.anchors),
            deterministic=args.deterministic,
            packed_cuda_graph_max_requests=args.packed_graph_max_requests,
            packed_cuda_graph_max_tokens=(
                args.packed_graph_max_tokens
                or max(1024, 1 << (args.max_branches * args.block_size
                                  + args.block_size - 1).bit_length())
            ),
            dllm_algorithm_config=str(config_path),
        )
    from transformers import AutoConfig

    model_vocab_size = AutoConfig.from_pretrained(
        args.model, trust_remote_code=True
    ).vocab_size
    service = StrictDraftService(backend, speculate=not args.no_speculate)
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(args.endpoint)
    print("READY " + args.endpoint, flush=True)
    try:
        while True:
            request = socket.recv_json()
            try:
                if request["op"] == "hello":
                    socket.send_json({
                        "fingerprint": backend.fingerprint,
                        "block_size": backend.block_size,
                        "vocab_size": model_vocab_size,
                    })
                elif request["op"] == "stats":
                    socket.send_json(service.metrics)
                elif request["op"] == "step":
                    reply, pending = service.step(request["items"])
                    socket.send_json(reply)
                    service.speculate(pending)
                else:
                    socket.send_json({"error": "unknown operation"})
            except Exception as error:
                print(f"FATAL {type(error).__name__}: {error}", flush=True)
                raise
            if args.metrics_file:
                with open(args.metrics_file, "w") as output:
                    json.dump(service.metrics, output, indent=2)
    finally:
        socket.close()
        context.term()
        backend.shutdown()


if __name__ == "__main__":
    main()
