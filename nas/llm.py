"""Pluggable LLM backends: Ollama (local), any OpenAI-compatible API, and a mock (offline testing)."""
import json
import os
import random
import time
import urllib.error
import urllib.request


class LLMError(Exception):
    pass


def _post(url, payload, headers, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


class OllamaLLM:
    def __init__(self, cfg):
        self.cfg = cfg
        self.url = cfg.get("base_url", "http://localhost:11434").rstrip("/") + "/api/chat"
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}

    def chat(self, system, user, temperature=None):
        opts = {"temperature": temperature if temperature is not None else self.cfg.get("temperature", 0.8),
                "num_ctx": self.cfg.get("num_ctx", 8192), "num_predict": self.cfg.get("max_tokens", 3000)}
        if self.cfg.get("num_gpu") is not None:
            opts["num_gpu"] = self.cfg["num_gpu"]  # 0 = keep the LLM entirely on CPU (leaves VRAM for training)
        if self.cfg.get("num_thread"):
            opts["num_thread"] = self.cfg["num_thread"]
        payload = {"model": self.cfg["model"], "stream": False, "options": opts,
                   "keep_alive": self.cfg.get("keep_alive", "30m"),
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        t0 = time.time()
        r = _retry(lambda: _post(self.url, payload, {}, self.cfg.get("request_timeout", 600)))
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += r.get("prompt_eval_count", 0)
        self.usage["completion_tokens"] += r.get("eval_count", 0)
        self.usage["seconds"] += time.time() - t0
        return r["message"]["content"]


class OpenAILLM:
    """Works with OpenAI, OpenRouter, Groq, Together, GitHub Models, vLLM, LM Studio, etc."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.url = cfg.get("base_url", "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        key = os.environ.get(cfg.get("api_key_env", "OPENAI_API_KEY"), "")
        if not key:
            raise LLMError(f"environment variable {cfg.get('api_key_env', 'OPENAI_API_KEY')} is not set")
        self.headers = {"Authorization": f"Bearer {key}"}
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}

    def chat(self, system, user, temperature=None):
        payload = {"model": self.cfg["model"], "max_tokens": self.cfg.get("max_tokens", 3000),
                   "temperature": temperature if temperature is not None else self.cfg.get("temperature", 0.8),
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        t0 = time.time()
        r = _retry(lambda: _post(self.url, payload, self.headers, self.cfg.get("request_timeout", 600)))
        u = r.get("usage", {})
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += u.get("prompt_tokens", 0)
        self.usage["completion_tokens"] += u.get("completion_tokens", 0)
        self.usage["seconds"] += time.time() - t0
        return r["choices"][0]["message"]["content"]


class MockLLM:
    """Offline backend: returns a random architecture from the random-search space. For tests and dry runs."""

    def __init__(self, cfg):
        self.rng = random.Random(cfg.get("seed", 0))
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}

    def chat(self, system, user, temperature=None):
        from .space import random_arch_code

        self.usage["calls"] += 1
        idea, code = random_arch_code(self.rng)
        return f"IDEA: {idea}\n```python\n{code}```"


def _retry(fn, tries=4):
    for i in range(tries):
        try:
            return fn()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            body = ""
            if isinstance(e, urllib.error.HTTPError):
                body = e.read().decode(errors="ignore")[:300]
                if e.code in (400, 401, 403, 404):
                    raise LLMError(f"HTTP {e.code}: {body}")
            if i == tries - 1:
                raise LLMError(f"LLM request failed after {tries} tries: {e} {body}")
            time.sleep(5 * 2 ** i)


def make_llm(cfg):
    backend = cfg.get("backend", "ollama")
    if backend == "ollama":
        return OllamaLLM(cfg)
    if backend == "openai":
        return OpenAILLM(cfg)
    if backend == "mock":
        return MockLLM(cfg)
    raise ValueError(f"unknown llm backend {backend}")
