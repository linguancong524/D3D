import importlib.util
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "d3d_sglang_anchor_sweep", ROOT / "scripts/sglang/sweep_anchors.py"
)
SWEEP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SWEEP)


class AnchorSweepTests(unittest.TestCase):
    def test_finished_process_is_not_signalled(self):
        process = Mock()
        process.poll.return_value = 0
        with patch.object(SWEEP.psutil, "Process") as lookup:
            SWEEP.stop_tree(process)
        lookup.assert_not_called()

    def test_only_owned_process_tree_is_stopped(self):
        process = Mock(pid=1234)
        process.poll.return_value = None
        parent, child = Mock(), Mock()
        parent.children.return_value = [child]
        with patch.object(SWEEP.psutil, "Process", return_value=parent) as lookup:
            with patch.object(SWEEP.psutil, "wait_procs", return_value=([parent, child], [])):
                SWEEP.stop_tree(process)
        lookup.assert_called_once_with(1234)
        parent.children.assert_called_once_with(recursive=True)
        parent.terminate.assert_called_once()
        child.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=10)


if __name__ == "__main__":
    unittest.main()
