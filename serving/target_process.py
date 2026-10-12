"""Persistent target worker. Only CPU request metadata crosses the pipe."""
import multiprocessing as mp
import traceback
import torch
from serving.backend import Runner, Sequence, SamplingParams
from common.runtime import resolve_model_path


def _worker(pipe, path, pages):
    try:
        torch.set_num_threads(4)
        with torch.inference_mode():
            runner = Runner(resolve_model_path(path), 'cuda:0', pages=pages)
            sequences = {}
            pipe.send(('ok', None))
            while True:
                command, payload, updates = pipe.recv()
                for kind, key, length in updates:
                    seq = sequences[key]
                    if kind == 'release':
                        runner.release(seq)
                        del sequences[key]
                    else:
                        runner.trim(seq, length)
                if command == 'close':
                    break
                if command == 'prefill':
                    group = []
                    for key, ids in payload:
                        if key in sequences:
                            raise RuntimeError(f'Duplicate sequence {key}')
                        seq = Sequence(ids, SamplingParams(temperature=0))
                        seq.prefix = len(ids)
                        sequences[key] = seq
                        group.append(seq)
                    result = runner.prefill(group)
                elif command == 'forward':
                    states, blocks = payload
                    group = []
                    for key, prefix in states:
                        seq = sequences[key]
                        seq.prefix = prefix
                        group.append(seq)
                    result = runner.forward(group, blocks).cpu().tolist()
                elif command == 'reset_stats':
                    if sequences:
                        raise RuntimeError('Previous generation left target sequences allocated')
                    torch.cuda.synchronize(0)
                    torch.cuda.reset_peak_memory_stats(0)
                    result = None
                elif command == 'stats':
                    torch.cuda.synchronize(0)
                    result = {'peak_allocated_gb': torch.cuda.max_memory_allocated(0)/1e9,
                              'active_sequences': len(sequences),
                              'free_pages': len(runner.pool.free_block_ids)}
                else:
                    raise ValueError(f'Unknown target command: {command}')
                pipe.send(('ok', result))
    except EOFError:
        pass
    except BaseException:
        try:
            pipe.send(('error', traceback.format_exc()))
        except (BrokenPipeError, OSError):
            pass
        raise
    finally:
        pipe.close()


class TargetProcess:
    def __init__(self, path, pages):
        context = mp.get_context('spawn')
        self.pipe, child = context.Pipe()
        self.process = context.Process(target=_worker, args=(child, path, pages))
        self.pending = True  # startup response
        self.updates = []
        self.closed = False
        self.process.start()
        child.close()

    def receive(self):
        if not self.pending:
            raise RuntimeError('No outstanding target request')
        try:
            while not self.pipe.poll(1):
                if not self.process.is_alive():
                    raise RuntimeError(f'Target worker exited: {self.process.exitcode}')
            kind, result = self.pipe.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError(f'Target pipe closed; worker exit={self.process.exitcode}') from exc
        finally:
            self.pending = False
        if kind == 'error':
            raise RuntimeError(f'Target worker failed:\n{result}')
        return result

    def submit(self, command, payload=None):
        if self.closed or self.pending:
            raise RuntimeError('Target is closed or already has an outstanding request')
        self.pipe.send((command, payload, self.updates))
        self.updates = []
        self.pending = True

    def request(self, command, payload=None):
        self.submit(command, payload)
        return self.receive()

    def submit_prefill(self, seqs):
        self.submit('prefill', [(s.seq_id, s.token_ids) for s in seqs])

    def prefill(self, seqs):
        self.submit_prefill(seqs)
        return self.receive()

    def submit_forward(self, seqs, blocks):
        self.submit('forward', ([(s.seq_id, s.prefix) for s in seqs], blocks))

    def release(self, seq):
        self.updates.append(('release', seq.seq_id, None))

    def trim(self, seq, length):
        self.updates.append(('trim', seq.seq_id, length))

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.process.is_alive() and not self.pending:
                self.pipe.send(('close', None, self.updates))
                self.process.join(timeout=10)
        except (BrokenPipeError, OSError):
            pass
        finally:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=10)
            self.pipe.close()
