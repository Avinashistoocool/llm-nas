"""Static checks and code extraction for LLM-generated architectures."""
import ast
import re

ALLOWED_IMPORTS = {"torch", "torch.nn", "torch.nn.functional", "math", "typing", "collections", "functools", "itertools"}
FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input", "breakpoint", "globals", "locals",
    "vars", "os", "sys", "subprocess", "shutil", "socket", "pathlib", "importlib", "builtins",
    "setattr", "delattr", "memoryview",
}
FORBIDDEN_ATTRS = {"load", "save", "hub", "system", "popen", "remove", "rmdir", "unlink", "cuda", "distributed",
                   "__subclasses__", "__globals__", "__builtins__", "__code__"}


class ValidationError(Exception):
    pass


def extract_code(text):
    """Return (idea, code) from an LLM response. Takes the longest fenced python block (or the raw text)."""
    idea = ""
    m = re.search(r"IDEA\s*:\s*(.+)", text)
    if m:
        idea = m.group(1).strip()[:300]
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    if not blocks:
        # tolerate an unterminated final block
        m2 = re.search(r"```(?:python|py)?\s*\n(.*)$", text, flags=re.S)
        blocks = [m2.group(1)] if m2 else [text]
    blocks = [b for b in blocks if "class Net" in b] or blocks
    code = max(blocks, key=len).strip() + "\n"
    code = _ensure_imports(code)
    if not idea:
        m3 = re.search(r'"""(.*?)"""', code, flags=re.S)
        if m3:
            idea = " ".join(m3.group(1).split())[:300]
    return idea, code


def _ensure_imports(code):
    """Small models often drop the import header; add the standard aliases if they're used but missing."""
    header = []
    if not re.search(r"^\s*import torch\s*$", code, flags=re.M):
        header.append("import torch")
    if "nn." in code and not re.search(r"import torch\.nn as nn|from torch import nn", code):
        header.append("import torch.nn as nn")
    if "F." in code and not re.search(r"import torch\.nn\.functional as F", code):
        header.append("import torch.nn.functional as F")
    if "math." in code and not re.search(r"^\s*import math", code, flags=re.M):
        header.append("import math")
    return ("\n".join(header) + "\n\n" + code) if header else code


def static_check(code):
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValidationError(f"SyntaxError: {e.msg} (line {e.lineno})")
    has_net = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in ALLOWED_IMPORTS:
                    raise ValidationError(f"import of '{a.name}' is not allowed (allowed: {sorted(ALLOWED_IMPORTS)})")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "") not in ALLOWED_IMPORTS:
                raise ValidationError(f"import from '{node.module}' is not allowed (allowed: {sorted(ALLOWED_IMPORTS)})")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise ValidationError(f"use of '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            raise ValidationError(f"attribute '.{node.attr}' is not allowed")
        elif isinstance(node, ast.ClassDef) and node.name == "Net":
            has_net = True
        elif isinstance(node, (ast.While,)):
            raise ValidationError("while loops are not allowed in model code")
    if not has_net:
        raise ValidationError("code must define `class Net(nn.Module)`")
    return True


def build_model(code, num_classes=10):
    """Exec validated code in a fresh namespace and instantiate Net. Only call inside a sandbox subprocess."""
    static_check(code)
    ns = {"__name__": "candidate"}
    exec(compile(code, "<candidate>", "exec"), ns)
    Net = ns["Net"]
    try:
        return Net(num_classes=num_classes)
    except TypeError:
        return Net()
