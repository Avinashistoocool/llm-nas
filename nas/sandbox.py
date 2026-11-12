"""Runs candidate code in an isolated subprocess with a hard timeout."""
import json
import os
import subprocess
import sys
import tempfile

from .config import ROOT


def run_job(job, timeout):
    with tempfile.TemporaryDirectory(prefix="nasjob_") as td:
        job = dict(job)
        job_path = os.path.join(td, "job.json")
        job["result_path"] = os.path.join(td, "result.json")
        with open(job_path, "w", encoding="utf-8") as f:
            json.dump(job, f)
        env = dict(os.environ)
        env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONUNBUFFERED"] = "1"
        if job["mode"] == "check":
            env["CUDA_VISIBLE_DEVICES"] = ""
        flags = subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" and job["mode"] == "check" else 0
        try:
            p = subprocess.run([sys.executable, "-m", "nas.worker", job_path], cwd=ROOT, env=env,
                               capture_output=True, text=True, timeout=timeout, creationflags=flags)
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": f"exceeded hard timeout of {timeout}s"}
        if not os.path.exists(job["result_path"]):
            err = (p.stderr or p.stdout or "").strip()[-1500:]
            return {"status": "crash", "error": f"worker exited with code {p.returncode}: {err}"}
        with open(job["result_path"], "r", encoding="utf-8") as f:
            return json.load(f)
