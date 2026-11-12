"""Fixed training recipe shared by every candidate (the LLM only controls the architecture)."""
import math
import time

import torch
import torch.nn.functional as F

from .data import GPULoader


class TooSlow(Exception):
    pass


class Diverged(Exception):
    pass


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model, device="cpu"):
    from torch.utils.flop_counter import FlopCounterMode

    model.eval()
    x = torch.randn(1, 3, 32, 32, device=device)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops()


@torch.no_grad()
def evaluate(model, x, y, device, batch_size=1000):
    model.eval()
    loader = GPULoader(x, y, batch_size, shuffle=False, aug=False, device=device)
    correct = 0
    for xb, yb in loader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            out = model(xb)
        correct += (out.float().argmax(1) == yb).sum().item()
    return correct / len(x)


def train_eval(model, splits, hp, device="cuda", seed=0, eval_sets=("val",)):
    torch.backends.cudnn.benchmark = True
    x, y = splits["train_x"], splits["train_y"]
    subset = hp.get("train_subset") or 0
    if 0 < subset < len(x):
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(x), generator=g)[:subset]
        x, y = x[idx], y[idx]
    loader = GPULoader(x, y, hp["batch_size"], shuffle=True, aug=True, device=device, seed=seed, drop_last=True)
    model.to(device)

    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (decay if p.ndim >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": hp["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
        lr=hp["lr"],
    )
    total = hp["epochs"] * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=hp["lr"], total_steps=total, pct_start=0.25)

    t0 = time.time()
    last_loss = float("nan")
    for epoch in range(hp["epochs"]):
        model.train()
        running, nb = 0.0, 0
        for xb, yb in loader:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                out = model(xb)
            loss = F.cross_entropy(out.float(), yb, label_smoothing=hp.get("label_smoothing", 0.0))
            if not math.isfinite(loss.item()):
                raise Diverged(f"non-finite loss at epoch {epoch}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            running += loss.item()
            nb += 1
        last_loss = running / max(nb, 1)
        if time.time() - t0 > hp["max_seconds"]:
            raise TooSlow(f"exceeded {hp['max_seconds']}s after epoch {epoch + 1}/{hp['epochs']}")
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    train_time = time.time() - t0

    res = {"train_loss": last_loss, "train_time": train_time,
           "images_per_sec": hp["epochs"] * len(loader) * hp["batch_size"] / train_time}
    for name in eval_sets:
        res[f"{name}_acc"] = evaluate(model, splits[f"{name}_x"], splits[f"{name}_y"], device)
    return res
