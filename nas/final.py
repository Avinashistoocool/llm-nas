"""Fully train the top-k architectures of one or more runs and report test accuracy (the test set is used only here).

    python -m nas.final --runs llm_qwen3b random
"""
import argparse
import json
import os

from .config import ROOT, load_config, parse_override
from .sandbox import run_job
from .search import Archive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "cifar10.yaml"))
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, dict(parse_override(s) for s in args.set))
    k = args.top_k or cfg["final"]["top_k"]
    out_path = os.path.join(ROOT, "runs", "final_results.json")
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for run in args.runs:
        A = Archive(os.path.join(ROOT, "runs", run))
        for e in A.best(k):
            key = f"{run}#{e['id']}"
            if key in results and results[key].get("status") == "ok":
                print("skip (done)", key)
                continue
            code_path = os.path.join(A.code_dir, f"{e['id']:04d}_ok.py")
            print(f"final training {key} proxy_val={e['val_acc']:.4f} params={e['params']:,} ...", flush=True)
            res = run_job({"mode": "train", "code_path": code_path, "hp": cfg["final"], "seed": cfg["seed"],
                           "data_root": cfg["task"]["data_root"], "val_size": cfg["task"]["val_size"],
                           "num_classes": cfg["task"]["num_classes"], "device": cfg["device"],
                           "eval_sets": ["val", "test"],
                           "save_weights": os.path.join(A.run_dir, f"final_{e['id']:04d}.pt")},
                          timeout=cfg["final"]["timeout"])
            res.update(run=run, id=e["id"], origin=e["origin"], idea=e.get("idea", ""), proxy_val_acc=e["val_acc"],
                       params=e["params"], mflops=e.get("mflops"))
            results[key] = res
            print(json.dumps(res), flush=True)
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
