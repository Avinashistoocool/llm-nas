"""Evolutionary search loop with pluggable proposal strategies.

Strategies
  llm         - LLM mutation/crossover of tournament-selected parents, with score feedback (main method)
  llm_noscore - ablation: same, but the prompt hides all accuracies (LLM sees only code)
  zeroshot    - LLM writes architectures from scratch, no feedback
  random      - random sampling from a hand-designed space (classic NAS baseline)
"""
import argparse
import hashlib
import json
import os
import random
import time

from . import prompts
from .config import ROOT, load_config, parse_override
from .llm import LLMError, make_llm
from .sandbox import run_job
from .space import random_arch_code, seed_archs
from .validator import ValidationError, extract_code, static_check


class Archive:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.path = os.path.join(run_dir, "archive.jsonl")
        self.code_dir = os.path.join(run_dir, "models")
        os.makedirs(self.code_dir, exist_ok=True)
        self.entries = []
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                self.entries = [json.loads(l) for l in f if l.strip()]

    def add(self, e):
        e["id"] = len(self.entries)
        e["time"] = time.time()
        self.entries.append(e)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({k: v for k, v in e.items() if k != "code"}) + "\n")
        with open(os.path.join(self.code_dir, f"{e['id']:04d}_{e['status']}.py"), "w", encoding="utf-8") as f:
            f.write(f'# id={e["id"]} origin={e["origin"]} status={e["status"]} val_acc={e.get("val_acc")}\n'
                    f'# idea: {e.get("idea", "")}\n' + e["code"])
        return e

    def code(self, e):
        if "code" not in e:
            fn = os.path.join(self.code_dir, f"{e['id']:04d}_{e['status']}.py")
            with open(fn, "r", encoding="utf-8") as f:
                e["code"] = "".join(f.readlines()[2:])
        return e["code"]

    @property
    def ok(self):
        return [e for e in self.entries if e["status"] == "ok"]

    @property
    def n_trained(self):
        return sum(1 for e in self.entries if e.get("trained"))

    def best(self, k):
        return sorted(self.ok, key=lambda e: e["fitness"], reverse=True)[:k]

    def hashes(self):
        return {e["hash"] for e in self.entries}


def code_hash(code):
    norm = "\n".join(l.split("#")[0].rstrip() for l in code.splitlines() if l.strip() and not l.strip().startswith("#"))
    return hashlib.sha1(norm.encode()).hexdigest()[:12]


class Search:
    def __init__(self, cfg, run_dir, strategy, log=print):
        self.cfg = cfg
        self.strategy = strategy
        self.archive = Archive(run_dir)
        self.rng = random.Random(cfg["seed"] + len(self.archive.entries))
        self.log = log
        self.llm = None if strategy == "random" else make_llm(cfg["llm"])
        c = cfg["constraints"]
        self.limits = {"max_params": c["max_params"], "max_mflops": c["max_mflops"]}
        self.sys_prompt = prompts.SYSTEM.format(max_params=c["max_params"], max_mflops=c["max_mflops"],
                                                epochs=cfg["proxy"]["epochs"], subset=cfg["proxy"]["train_subset"])
        self.n_proposals = len(self.archive.entries)
        self.started = time.time()

    # ---------------------------------------------------------------- evaluation
    def check(self, code):
        """Static + CPU dry-run checks. Returns result dict (status ok/invalid/error...)."""
        try:
            static_check(code)
        except ValidationError as e:
            return {"status": "invalid", "error": str(e)}
        return run_job({"mode": "check", "code_path": self._tmp_code(code), "num_classes": self.cfg["task"]["num_classes"],
                        "limits": self.limits}, timeout=self.cfg["search"]["check_timeout"])

    def train(self, code, hp_key="proxy"):
        hp = self.cfg[hp_key]
        return run_job({"mode": "train", "code_path": self._tmp_code(code), "hp": hp, "seed": self.cfg["seed"],
                        "data_root": self.cfg["task"]["data_root"], "val_size": self.cfg["task"]["val_size"],
                        "num_classes": self.cfg["task"]["num_classes"], "device": self.cfg["device"],
                        "gpu_mem_fraction": self.cfg.get("gpu_mem_fraction")}, timeout=hp["timeout"])

    def _tmp_code(self, code):
        import threading
        path = os.path.join(self.archive.run_dir, f"_candidate_{threading.get_ident()}.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        return path

    def fitness(self, res):
        pen = self.cfg["search"].get("param_penalty", 0.0)
        return res["val_acc"] - pen * res["params"] / self.cfg["constraints"]["max_params"]

    def evaluate_and_record(self, origin, idea, code, parents=(), repairs=0, llm_seconds=0.0):
        h = code_hash(code)
        entry = {"origin": origin, "idea": idea, "code": code, "parents": list(parents), "hash": h,
                 "repairs": repairs, "llm_seconds": round(llm_seconds, 1), "strategy": self.strategy}
        if h in self.archive.hashes():
            entry.update(status="duplicate", error="identical to an earlier candidate")
            return self.archive.add(entry)
        chk = self.check(code)
        entry.update({k: chk[k] for k in ("params", "mflops") if k in chk})
        if chk["status"] != "ok":
            entry.update(status=chk["status"], error=chk.get("error", ""))
            return self.archive.add(entry)
        res = self.train(code)
        entry["trained"] = True
        entry.update({k: v for k, v in res.items() if k not in ("params",)})
        if res["status"] == "ok":
            entry["fitness"] = self.fitness({**res, "params": entry["params"]})
        return self.archive.add(entry)

    # ---------------------------------------------------------------- proposals
    def propose(self):
        """Returns (origin, idea, code, parents, repairs, llm_seconds) or None."""
        if self.strategy == "random":
            idea, code = random_arch_code(self.rng)
            return "random", idea, code, [], 0, 0.0
        if self.strategy == "zeroshot":
            user = prompts.ZERO_SHOT.format(k=self.rng.randint(0, 10**6), max_params=self.limits["max_params"],
                                            max_mflops=self.limits["max_mflops"])
            return self._ask("zeroshot", user, [])
        pool = self.archive.ok
        if not pool:
            return None
        hide = self.strategy == "llm_noscore"
        board = self.archive.best(self.cfg["search"]["leaderboard_size"])
        board_txt = prompts.leaderboard_text(board) if not hide else "\n".join(
            f"{i}. {e.get('idea', '')[:160]}" for i, e in enumerate(board, 1))
        fails = [e for e in self.archive.entries[-12:] if e["status"] not in ("ok", "duplicate")][-4:]
        fails_txt = prompts.failures_text(fails)
        S = self.cfg["search"]
        if len(pool) >= 2 and self.rng.random() < S["crossover_prob"]:
            a, b = self._tournament(pool), self._tournament(pool)
            if a["id"] != b["id"]:
                user = prompts.CROSSOVER.format(leaderboard=board_txt, a_acc=a["val_acc"], a_params=a["params"],
                                                a_code=self.archive.code(a), b_acc=b["val_acc"], b_params=b["params"],
                                                b_code=self.archive.code(b), failures=fails_txt, **self._lim())
                if hide:
                    user = _strip_scores(user)
                return self._ask("crossover", user, [a["id"], b["id"]])
        p = self._tournament(pool)
        user = prompts.MUTATE.format(leaderboard=board_txt, parent_acc=p["val_acc"], parent_params=p["params"],
                                     parent_mflops=p.get("mflops", 0), parent_code=self.archive.code(p),
                                     failures=fails_txt, explore_hint=self.rng.choice(prompts.EXPLORE_HINTS), **self._lim())
        if hide:
            user = _strip_scores(user)
        return self._ask("mutate", user, [p["id"]])

    def _lim(self):
        return {"max_params": self.limits["max_params"], "max_mflops": self.limits["max_mflops"]}

    def _tournament(self, pool):
        k = min(self.cfg["search"]["tournament_size"], len(pool))
        return max(self.rng.sample(pool, k), key=lambda e: e["fitness"])

    def _ask(self, origin, user, parents):
        t0 = time.time()
        text = self.llm.chat(self.sys_prompt, user)
        idea, code = extract_code(text)
        repairs = 0
        for _ in range(self.cfg["search"]["max_repairs"]):
            chk = self.check(code)
            if chk["status"] == "ok":
                break
            repairs += 1
            self.log(f"    repair #{repairs}: {chk['status']}: {str(chk.get('error', ''))[:120]}")
            text = self.llm.chat(self.sys_prompt, prompts.REPAIR.format(error=str(chk.get("error", ""))[:1500], code=code),
                                 temperature=0.3)
            new_idea, code = extract_code(text)
            idea = idea or new_idea
        return origin, idea, code, parents, repairs, time.time() - t0

    # ---------------------------------------------------------------- main loop
    def seed(self):
        if self.archive.entries or self.strategy in ("random", "zeroshot"):
            return
        for name, idea, code in seed_archs():
            e = self.evaluate_and_record(f"seed:{name}", idea, code)
            self._report(e)

    def run(self):
        """Asynchronous steady-state evolution: a producer thread asks the LLM (CPU) for the next candidate while
        the main thread trains the current one (GPU). Proposals may therefore lag the archive by one candidate."""
        import queue
        import threading

        S = self.cfg["search"]
        self.seed()
        max_props = S["budget"] * S.get("max_proposal_factor", 4)
        q = queue.Queue(maxsize=max(1, S.get("prefetch", 1)))
        stop = threading.Event()

        def producer():
            while not stop.is_set():
                self.n_proposals += 1
                try:
                    prop = self.propose()
                except LLMError as e:
                    self.log(f"LLM error: {e}; sleeping 30s")
                    time.sleep(30)
                    continue
                except Exception as e:  # noqa: BLE001
                    q.put(e)
                    return
                while not stop.is_set():
                    try:
                        q.put(prop, timeout=1)
                        break
                    except queue.Full:
                        pass

        th = threading.Thread(target=producer, daemon=True)
        th.start()
        try:
            while self.archive.n_trained < S["budget"] and len(self.archive.entries) < max_props:
                prop = q.get()
                if isinstance(prop, Exception):
                    raise prop
                if prop is None:
                    raise RuntimeError("no valid parents in archive (all seeds failed?)")
                origin, idea, code, parents, repairs, llm_s = prop
                e = self.evaluate_and_record(origin, idea, code, parents, repairs, llm_s)
                self._report(e)
        finally:
            stop.set()
        self.save_summary()

    def _report(self, e):
        best = self.archive.best(1)
        b = f"{best[0]['val_acc']:.4f}" if best else "-"
        acc = f"{e['val_acc']:.4f}" if e.get("val_acc") is not None else "  -   "
        self.log(f"[{self.archive.n_trained:3d}/{self.cfg['search']['budget']}] #{e['id']:<3d} {e['origin']:<16s} "
                 f"{e['status']:<9s} acc={acc} params={e.get('params', 0):>9,} "
                 f"t={e.get('train_time', 0):5.1f}s llm={e.get('llm_seconds', 0):5.1f}s best={b} | {e.get('idea', '')[:70]}")
        if e["status"] not in ("ok",) and e.get("error"):
            self.log(f"      -> {str(e['error']).splitlines()[0][:150]}")

    def save_summary(self):
        A = self.archive
        st = {}
        for e in A.entries:
            st[e["status"]] = st.get(e["status"], 0) + 1
        summary = {"strategy": self.strategy, "n_entries": len(A.entries), "n_trained": A.n_trained, "status_counts": st,
                   "best": [{k: e.get(k) for k in ("id", "origin", "val_acc", "params", "mflops", "idea")} for e in A.best(5)],
                   "llm_usage": getattr(self.llm, "usage", None), "wall_seconds": time.time() - self.started}
        with open(os.path.join(A.run_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary


def _strip_scores(text):
    import re
    text = re.sub(r"val_acc=[0-9.]+,?\s*", "", text)
    return text.replace("(higher val_acc is better)", "")


def main():
    ap = argparse.ArgumentParser(description="LLM-guided neural architecture search")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "cifar10.yaml"))
    ap.add_argument("--strategy", default="llm", choices=["llm", "llm_noscore", "zeroshot", "random"])
    ap.add_argument("--run", required=True, help="run name (directory under runs/); re-running resumes")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. search.budget=50 llm.backend=mock")
    args = ap.parse_args()
    cfg = load_config(args.config, dict(parse_override(s) for s in args.set))
    run_dir = os.path.join(ROOT, "runs", args.run)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"strategy": args.strategy, **cfg}, f, indent=2)
    logf = open(os.path.join(run_dir, "log.txt"), "a", encoding="utf-8")

    def log(msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    log(f"=== run={args.run} strategy={args.strategy} backend={cfg['llm']['backend']}:{cfg['llm']['model']} ===")
    s = Search(cfg, run_dir, args.strategy, log)
    s.run()
    log(json.dumps(s.save_summary(), indent=2))


if __name__ == "__main__":
    main()
