"""Subprocess entry point: `python -m nas.worker <job.json>`. Writes result JSON next to the job file.

Modes:
  check - CPU only: build model, forward a dummy batch, count params and FLOPs.
  train - GPU: proxy/final training with the fixed recipe.
"""
import json
import os
import sys
import time
import traceback


def _limit_threads():
    import torch
    torch.set_num_threads(int(os.environ.get("NAS_CPU_THREADS", "4")))


def run(job):
    import torch

    from .train import Diverged, TooSlow, count_flops, count_params, train_eval
    from .validator import ValidationError, build_model

    _limit_threads()
    code = open(job["code_path"], "r", encoding="utf-8").read()
    res = {"status": "ok"}
    try:
        torch.manual_seed(job.get("seed", 0))
        model = build_model(code, job.get("num_classes", 10))
        params = count_params(model)
        res["params"] = params
        if job["mode"] == "check":
            model.eval()
            with torch.no_grad():
                out = model(torch.randn(2, 3, 32, 32))
            if not isinstance(out, torch.Tensor) or tuple(out.shape) != (2, job.get("num_classes", 10)):
                shape = tuple(out.shape) if isinstance(out, torch.Tensor) else type(out).__name__
                raise ValidationError(f"forward must return logits of shape (B, {job.get('num_classes', 10)}), got {shape}")
            res["mflops"] = count_flops(model) / 1e6
            lim = job.get("limits", {})
            if lim.get("max_params") and params > lim["max_params"]:
                raise ValidationError(f"too many parameters: {params:,} > limit {lim['max_params']:,}")
            if lim.get("max_mflops") and res["mflops"] > lim["max_mflops"]:
                raise ValidationError(f"too many FLOPs: {res['mflops']:.0f} MFLOPs > limit {lim['max_mflops']} MFLOPs")
            # trial backward pass catches in-place/autograd bugs before burning GPU time
            model.train()
            model(torch.randn(4, 3, 32, 32)).float().sum().backward()
        else:
            from .data import get_splits

            frac = job.get("gpu_mem_fraction")
            if frac and torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(frac)
            splits = get_splits(job["data_root"], job.get("val_size", 5000))
            res.update(train_eval(model, splits, job["hp"], device=job.get("device", "cuda"),
                                  seed=job.get("seed", 0), eval_sets=tuple(job.get("eval_sets", ["val"]))))
            if torch.cuda.is_available():
                res["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
            if job.get("save_weights"):
                torch.save(model.state_dict(), job["save_weights"])
    except ValidationError as e:
        res = {**res, "status": "invalid", "error": str(e)}
    except TooSlow as e:
        res = {**res, "status": "too_slow", "error": str(e)}
    except Diverged as e:
        res = {**res, "status": "diverged", "error": str(e)}
    except torch.cuda.OutOfMemoryError as e:
        res = {**res, "status": "oom", "error": "CUDA out of memory (model too large for memory budget)"}
    except Exception as e:  # noqa: BLE001 - any candidate bug is a reportable failure
        lines = code.splitlines()
        frames = [f for f in traceback.extract_tb(e.__traceback__) if f.filename == "<candidate>"]
        where = "\n".join(f"  candidate line {f.lineno} in {f.name}: {lines[f.lineno - 1].strip() if 0 < f.lineno <= len(lines) else ''}"
                          for f in frames[-3:])
        res = {**res, "status": "error", "error": f"{type(e).__name__}: {str(e)[:600]}\n{where}"}
    return res


def main():
    job_path = sys.argv[1]
    with open(job_path, "r", encoding="utf-8") as f:
        job = json.load(f)
    t0 = time.time()
    res = run(job)
    res["wall_time"] = time.time() - t0
    with open(job["result_path"], "w", encoding="utf-8") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
