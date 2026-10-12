import ast
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _module_functions(tree):
    return {
        node.name: {arg.arg for arg in node.args.args + node.args.kwonlyargs}
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


class CallSignatureTests(unittest.TestCase):
    """Keyword arguments at internal call sites must match the definition."""

    def test_internal_keyword_calls_match_definitions(self):
        checked = 0
        for relative in (
            'single/eval_sync.py',
            'single/eval_async_packed.py',
            'serving/bench.py',
        ):
            with self.subTest(module=relative):
                tree = ast.parse((ROOT / relative).read_text())
                signatures = _module_functions(tree)
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                        continue
                    signature = signatures.get(node.func.id)
                    if signature is None:
                        continue
                    for keyword in node.keywords:
                        if keyword.arg is None:
                            continue
                        self.assertIn(
                            keyword.arg,
                            signature,
                            f'{relative}:{node.lineno}: {node.func.id}() does not accept '
                            f'keyword {keyword.arg!r}',
                        )
                        checked += 1
        self.assertGreater(checked, 0, 'no internal keyword calls were checked')


if __name__ == '__main__':
    unittest.main()
