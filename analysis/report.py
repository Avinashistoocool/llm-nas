"""Plots + markdown report comparing runs.

    python -m analysis.report --runs llm_qwen3b random
"""
import argparse
import json
import os
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(run):
    with open(os.path.join(ROOT, "runs", run, "archive.jsonl")) as f:
        return [json.loads(l) for l in f if l.strip()]


def best_so_far(entries):
    xs, ys, best, n = [], [], 0.0, 0
    for e in entries:
        if not e.get("trained"):
            continue
        n += 1
        if e["status"] == "ok":
            best = max(best, e["val_acc"])
        xs.append(n)
        ys.append(best)
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "analysis"))
    args = ap.parse_args()
    data = {r: load(r) for r in args.runs}

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for r, es in data.items():
        xs, ys = best_so_far(es)
        ax[0].plot(xs, ys, label=r, lw=2)
        ok = [e for e in es if e["status"] == "ok" and not e["origin"].startswith("seed")]
        ax[1].scatter([e["params"] / 1e3 for e in ok], [e["val_acc"] for e in ok], label=r, alpha=0.7, s=25)
    seeds = [e for es in data.values() for e in es if e["origin"].startswith("seed") and e["status"] == "ok"]
    for e in {e["origin"]: e for e in seeds}.values():
        ax[1].scatter(e["params"] / 1e3, e["val_acc"], marker="*", s=180, c="k")
        ax[1].annotate(e["origin"][5:], (e["params"] / 1e3, e["val_acc"]), fontsize=8, xytext=(4, -10), textcoords="offset points")
    ax[0].set(xlabel="# proxy-training attempts", ylabel="best proxy val acc", title="Search progress")
    ax[1].set(xlabel="params (K)", ylabel="proxy val acc", title="Accuracy vs. size (stars = hand-designed seeds)")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    os.makedirs(args.out, exist_ok=True)
    fig.savefig(os.path.join(args.out, "search_progress.png"), dpi=130)

    lines = ["# Results", "", "| run | proposals | proxy attempts | completed | completed proposals / proposals | best proxy val | median proxy val | best params |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r, es in data.items():
        ok = [e for e in es if e["status"] == "ok"]
        gen = [e for e in es if not e["origin"].startswith("seed")]
        gen_ok = [e for e in gen if e["status"] == "ok"]
        best = max(ok, key=lambda e: e["val_acc"]) if ok else None
        accs = sorted(e["val_acc"] for e in gen_ok)
        med = median(accs) if accs else float("nan")
        attempts = sum(1 for e in es if e.get("trained"))
        lines.append(f"| {r} | {len(gen)} | {attempts} | {len(ok)} | "
                     f"{len(gen_ok) / max(len(gen), 1):.0%} | {best['val_acc']:.4f} | {med:.4f} | {best['params']:,} |")
    lines += ["", "## Failure modes", ""]
    for r, es in data.items():
        st = {}
        for e in es:
            st[e["status"]] = st.get(e["status"], 0) + 1
        lines.append(f"- **{r}**: " + ", ".join(f"{k}={v}" for k, v in sorted(st.items())))
    fpath = os.path.join(ROOT, "runs", "final_results.json")
    if os.path.exists(fpath):
        fr = json.load(open(fpath))
        lines += ["", "## Finalist training (test evaluated after training for each finalist)", "",
                  "| run#id | origin | params | proxy val | full val | **test** | idea |", "|---|---|---|---|---|---|---|"]
        for k, v in sorted(fr.items(), key=lambda kv: -kv[1].get("test_acc", 0)):
            if v.get("status") == "ok":
                lines.append(f"| {k} | {v['origin']} | {v['params']:,} | {v['proxy_val_acc']:.4f} | {v['val_acc']:.4f} | "
                             f"**{v['test_acc']:.4f}** | {v['idea'][:90]} |")
    lines += ["", "![progress](search_progress.png)", ""]
    for r, es in data.items():
        ok = sorted([e for e in es if e["status"] == "ok"], key=lambda e: -e["val_acc"])[:5]
        lines += [f"## Top-5 of `{r}`", ""] + [f"{i}. `#{e['id']}` {e['val_acc']:.4f}, {e['params']:,} params ({e['origin']}): {e.get('idea', '')[:150]}"
                                              for i, e in enumerate(ok, 1)] + [""]
    with open(os.path.join(args.out, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
