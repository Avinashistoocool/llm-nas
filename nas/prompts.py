"""Prompt templates. The LLM acts as a mutation / crossover operator over PyTorch source code."""

SYSTEM = """You are an expert deep-learning researcher doing neural architecture search for CIFAR-10 (32x32 RGB images, 10 classes).
You write complete, self-contained PyTorch model code.

HARD RULES (violations are rejected automatically):
- Define `class Net(nn.Module)` whose `__init__(self, num_classes=10)` builds the model and whose `forward(x)` maps
  a (B, 3, 32, 32) float tensor to logits of shape (B, num_classes).
- Only import from: torch, torch.nn, torch.nn.functional, math, typing, collections, functools, itertools.
- No file/network/OS access, no `while` loops, no `.cuda()` / device handling, no pretrained weights, no training code.
- Parameter budget: <= {max_params:,} trainable parameters. Compute budget: <= {max_mflops} MFLOPs per image.
- The training recipe is FIXED and NOT under your control (AdamW, one-cycle LR, bf16 autocast, random crop+flip,
  label smoothing, {epochs} epochs on {subset} images). Improve accuracy ONLY through the architecture.

OUTPUT FORMAT (exactly):
IDEA: <one sentence describing what you changed and why>
```python
<the COMPLETE file: all imports, every helper function/class you use, and class Net. Never reference code you did not write in this block.>
```"""

MUTATE = """Here are the best architectures found so far (higher val_acc is better):

{leaderboard}

Below is the PARENT architecture (val_acc={parent_acc:.4f}, params={parent_params:,}, mflops={parent_mflops:.0f}):
```python
{parent_code}
```
{failures}
Propose ONE new architecture by modifying the parent. Make a meaningful, targeted change that you believe will raise
validation accuracy under the fixed short training budget (e.g. width/depth reallocation, block type, kernel sizes,
normalisation, activation, attention/SE, downsampling schedule, skip connections, stem/head design).
{explore_hint}
Remember: <= {max_params:,} params, <= {max_mflops} MFLOPs, output exactly the IDEA line and one python code block."""

CROSSOVER = """Here are the best architectures found so far (higher val_acc is better):

{leaderboard}

PARENT A (val_acc={a_acc:.4f}, params={a_params:,}):
```python
{a_code}
```

PARENT B (val_acc={b_acc:.4f}, params={b_params:,}):
```python
{b_code}
```
{failures}
Combine the strengths of A and B into ONE new architecture that should beat both.
Remember: <= {max_params:,} params, <= {max_mflops} MFLOPs, output exactly the IDEA line and one python code block."""

ZERO_SHOT = """Design a strong, novel CIFAR-10 architecture for this setting.
Idea seed #{k}: be creative and different from a standard ResNet if you can, while staying trainable in a few epochs.
Remember: <= {max_params:,} params, <= {max_mflops} MFLOPs, output exactly the IDEA line and one python code block."""

REPAIR = """Your previous code was rejected.

ERROR:
{error}

CODE:
```python
{code}
```
Fix the problem (if it is a budget violation, shrink widths/depth; if a name is undefined, define it in the file).
Output exactly the IDEA line and one python code block containing the COMPLETE file."""

EXPLORE_HINTS = [
    "Prefer a bold change this time (explore).",
    "Prefer a small, careful refinement this time (exploit).",
    "Try reallocating capacity between stages.",
    "Consider an efficiency trick so you can afford more width or depth within the budget.",
]


def leaderboard_text(entries):
    lines = []
    for i, e in enumerate(entries, 1):
        lines.append(f"{i}. val_acc={e['val_acc']:.4f} params={e['params']:,} mflops={e.get('mflops', 0):.0f} "
                     f"| {e.get('idea', '')[:160]}")
    return "\n".join(lines) if lines else "(none yet)"


def failures_text(entries):
    if not entries:
        return ""
    lines = [f"- {e.get('idea', '')[:100]} -> {e['status']}: {str(e.get('error', ''))[:160]}" for e in entries]
    return "\nRecent FAILED attempts (avoid repeating these mistakes):\n" + "\n".join(lines) + "\n"
