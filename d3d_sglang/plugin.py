def register():
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    from sglang.srt.speculative.spec_registry import CustomSpecAlgo

    class StrictRemoteAlgorithm(CustomSpecAlgo):
        def is_host_draft(self):
            return True

        def has_draft_kv(self):
            return False

        def carries_draft_hidden_states(self):
            return False

        def need_topk(self):
            return False

        def create_future_map(
            self, device, req_to_token_pool, needs_cpu_seq_lens=True,
            needs_confidence_relay=False,
        ):
            from sglang.srt.managers.overlap_utils import FutureMap

            return FutureMap(
                device, self, req_to_token_pool,
                needs_cpu_seq_lens, needs_confidence_relay,
            )

        def handle_server_args(self, server_args):
            if not server_args.decoupled_draft_endpoint:
                raise ValueError("D3D_REMOTE requires --decoupled-draft-endpoint")
            if server_args.speculative_draft_model_path is not None:
                raise ValueError("D3D_REMOTE must not load a Target-local draft model")
            if server_args.tp_size != 1 or server_args.dp_size != 1:
                raise ValueError("D3D_REMOTE currently supports TP1/DP1 only")
            if server_args.speculative_num_draft_tokens is None:
                server_args.speculative_num_draft_tokens = 16
            if server_args.speculative_num_draft_tokens < 2:
                raise ValueError("draft block must contain anchor and candidate tokens")
            server_args.speculative_eagle_topk = 1
            server_args.speculative_num_steps = (
                server_args.speculative_num_draft_tokens - 1
            )
            server_args.disable_overlap_schedule = True
            server_args.enable_mixed_chunk = False

    @SpeculativeAlgorithm.register(
        "D3D_REMOTE", supports_overlap=True, spec_class=StrictRemoteAlgorithm
    )
    def factory(server_args):
        from .worker import StrictRemoteWorker

        return StrictRemoteWorker
