import atexit
import time

import numpy as np
import zmq
from sglang.srt.speculative.decoupled.protocol import vocabulary_fingerprint
from sglang.srt.speculative.host_draft_worker import HostDraftWorker
from transformers import AutoTokenizer


class StrictRemoteWorker(HostDraftWorker):
    def __init__(self, server_args, gpu_id, ps, nccl_port, target_worker):
        super().__init__(server_args, gpu_id, ps, nccl_port, target_worker)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        timeout = int(server_args.decoupled_draft_timeout * 1000)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout)
        self.socket.connect(server_args.decoupled_draft_endpoint)
        tokenizer = AutoTokenizer.from_pretrained(
            server_args.tokenizer_path or server_args.model_path,
            trust_remote_code=server_args.trust_remote_code,
        )
        reply = self._rpc({"op": "hello"})
        if reply["fingerprint"] != vocabulary_fingerprint(tokenizer.get_vocab()):
            raise ValueError("remote Draft and Target tokenizer IDs differ")
        if reply["block_size"] != self.draft_token_num:
            raise ValueError("remote Draft and Target block sizes differ")
        if reply["vocab_size"] != self.model_runner.model_config.vocab_size:
            raise ValueError("remote Draft and Target model vocabularies differ")
        self.vocab_size = reply["vocab_size"]
        self.chain_mask = np.tril(
            np.ones((self.draft_token_num, self.draft_token_num), dtype=np.bool_)
        )
        self.metrics = {"remote_round_trips": 0, "remote_wait_seconds": 0.0}
        self.closed = False
        atexit.register(self.close)

    def _rpc(self, payload):
        self.socket.send_json(payload)
        reply = self.socket.recv_json()
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply

    def _prepare_draft_tokens(self, batch):
        items = []
        for request in batch.reqs:
            if request.sampling_params.top_k != 1:
                raise ValueError("D3D_REMOTE currently supports greedy decoding only")
            if not request.output_ids:
                raise RuntimeError("remote drafting requires a Target prefill token")
            items.append({
                "rid": request.rid,
                "prompt": list(request.origin_input_ids),
                "outputs": list(request.output_ids),
            })
        started = time.perf_counter()
        reply = self._rpc({"op": "step", "items": items})
        self.metrics["remote_wait_seconds"] += time.perf_counter() - started
        self.metrics["remote_round_trips"] += 1
        rows = []
        for request in batch.reqs:
            tokens = reply["proposals"][request.rid]["tokens"]
            if (
                len(tokens) != self.draft_token_num
                or tokens[0] != request.output_ids[-1]
                or any(not 0 <= token < self.vocab_size for token in tokens)
            ):
                raise RuntimeError("remote Draft returned an invalid proposal")
            rows.append(tokens)
        return (
            np.asarray(rows, dtype=np.int64).reshape(-1),
            np.tile(self.chain_mask, (len(rows), 1, 1)).reshape(-1),
        )

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.socket.close()
        self.context.term()

    def update_weights_from_disk(self, recv_req):
        return True, "Target-only update; remote Draft is independently configured"

    def update_weights_from_ipc(self, recv_req):
        return True, "Target-only update; remote Draft is independently configured"
