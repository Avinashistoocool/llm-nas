import pytest

from nas.search import _strip_scores, code_hash
from nas.space import random_arch_code, seed_archs
from nas.validator import ValidationError, build_model, extract_code, static_check
import random
import torch


def test_extract_code_takes_block_and_idea():
    idea, code = extract_code("IDEA: wider stem\n```python\nimport torch\nclass Net: pass\n```\ntrailing")
    assert idea == "wider stem"
    assert code.startswith("import torch") and "class Net" in code


def test_extract_code_unterminated_block():
    _, code = extract_code("IDEA: x\n```python\nclass Net:\n    pass\n")
    assert "class Net" in code


@pytest.mark.parametrize("bad", [
    "import os\nclass Net: pass",
    "from subprocess import run\nclass Net: pass",
    "class Net:\n    def f(self): open('x')",
    "import torch\nclass Net:\n    def f(self): torch.load('x')",
    "class Net:\n    def f(self):\n        while True: pass",
    "class Model: pass",
])
def test_static_check_rejects(bad):
    with pytest.raises(ValidationError):
        static_check(bad)


def test_seeds_build_and_forward():
    for _, _, code in seed_archs():
        m = build_model(code)
        assert m(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_random_space_builds():
    rng = random.Random(0)
    for _ in range(5):
        _, code = random_arch_code(rng)
        assert build_model(code)(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_code_hash_ignores_comments_and_blank_lines():
    assert code_hash("a = 1\n\n# hi\nb = 2  # x\n") == code_hash("a = 1\nb = 2\n")


def test_strip_scores():
    assert "val_acc" not in _strip_scores("PARENT (val_acc=0.7123, params=5)")
