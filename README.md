# llm-nas: LLM-guided neural architecture search

An LLM acts as the **mutation and crossover operator** in an evolutionary search over **raw PyTorch source code**. Each candidate is checked statically and with a CPU dry run (parameter/FLOP budget, forward and backward shapes), trained briefly in a sandboxed subprocess, and scored. The scores go back into the next prompt.

```
 archive (code + scores) ──► tournament select ──► LLM: mutate / crossover / repair ──► AST + CPU dry-run checks
        ▲                                                                                          │
        └──────────── proxy training on GPU (fixed recipe, subprocess, timeout) ◄─────────────────┘
```

Built for a **6 GB laptop GPU** that is shared with other jobs:
- **Local LLM on the CPU** (Ollama `qwen2.5-coder:3b`, `num_gpu: 0`). The GPU is left for training only.
- **Asynchronous pipeline:** the LLM writes candidate *n+1* while candidate *n* trains.
- **Whole dataset on the GPU as uint8**, with GPU-side crop/flip augmentation. A 5-epoch proxy on 10k images takes about 15–50 s.
- The LLM can't touch the training recipe (AdamW, one-cycle, bf16, label smoothing). Only the architecture changes.

## Strategies (for comparison)
| strategy | description |
|---|---|
| `llm` | LLM mutation (p=0.75) / crossover (p=0.25) of tournament-selected parents, with leaderboard, recent-failure feedback, and up to 2 repair rounds |
| `llm_noscore` | ablation: the same, but all accuracies are removed from prompts |
| `zeroshot` | LLM writes architectures from scratch with no feedback |
| `random` | random sampling from a hand-designed space (block type, widths, depths, kernel, act, SE, residual) |

## Usage
```powershell
# data: HF parquet copy of CIFAR-10 (cached to data/cifar10_uint8.pt on first use)
curl.exe -L -o data/train.parquet https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet
curl.exe -L -o data/test.parquet  https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet

ollama pull qwen2.5-coder:3b
python -m nas.search --run llm_qwen3b --strategy llm   --set search.budget=40
python -m nas.search --run random     --strategy random --set search.budget=40
python -m nas.final  --runs llm_qwen3b random           # full training of top-k; test set used ONLY here
python -m analysis.report --runs llm_qwen3b random      # analysis/RESULTS.md + search_progress.png

# any OpenAI-compatible API (OpenAI, OpenRouter, Groq, GitHub Models, vLLM, LM Studio...)
$env:OPENAI_API_KEY="..."
python -m nas.search --run llm_gpt --set llm.backend=openai llm.model=gpt-4o-mini llm.base_url=https://api.openai.com/v1

# offline smoke test (no LLM)
python -m nas.search --run _mock --set llm.backend=mock search.budget=5 proxy.epochs=1
python -m pytest -q tests
```
Runs resume automatically if you re-run the same `--run` name. All candidates (including failures) are saved under `runs/<run>/models/`.

## Layout
```
configs/cifar10.yaml   budget, constraints (<=1M params, <=600 MFLOPs), proxy/final recipes, LLM backend
nas/data.py            CIFAR-10 on GPU, fixed 45k/5k train/val split, GPU augmentation
nas/train.py           fixed training recipe, params/FLOPs counting
nas/validator.py       code extraction + AST safety checks (import allow-list, no I/O / exec / while)
nas/worker.py          subprocess entry point (check on CPU / train on GPU)
nas/sandbox.py         subprocess + hard timeout
nas/llm.py             Ollama / OpenAI-compatible / mock backends
nas/prompts.py         system, mutate, crossover, zero-shot, repair prompts
nas/space.py           3 hand-designed seeds + random search space
nas/search.py          archive, tournament selection, async evolution loop
nas/final.py           full training of top-k + test accuracy
analysis/report.py     plots and results table
```

## Notes and limitations
- The proxy (5 epochs, 10k images) is noisy and favours fast-converging models. `nas.final` re-trains the winners for 20 epochs on all 45k images to check that the ranking holds.
- The search uses the validation split only. The test set is evaluated once per finalist.
- The AST sandbox is a guard rail, not a security boundary. Only run code from LLMs you trust.

See `analysis/RESULTS.md` for results.

## Results (first run, 12 Nov 2026: 30 trained candidates per strategy, local qwen2.5-coder:3b on CPU)
| | LLM-guided (3B) | Random search |
|---|---|---|
| proposals / trained | 46 / 30 | 33 / 30 |
| valid-proposal rate | 59% | 85% |
| best proxy val acc | 0.667 (298K params) | 0.691 (628K params) |
| median proxy val acc | **0.598** | 0.569 |
| best **test** acc after 20-epoch full training | 89.3% (298K, ~75 s) | **92.7%** (801K, ~19 min) |

**Findings**
- **A 3B local model is too weak to beat random search** on best-found accuracy. Its median candidate is better (0.598 vs. 0.569) because it stays close to good parents. But only 2 of 27 children beat their parent, and it mostly repeated one idea ("add depthwise-separable conv"). Most failures were duplicates or broken code: 9 duplicates, 8 runtime errors, and 41 repair calls.
- **Mode collapse:** the LLM's ideas converged on a single mutation. Random search explored wider inverted-residual + SE models that use the parameter budget much better.
- **The LLM stayed near the cheap seeds.** It never used more than ~300K of the 1M-parameter budget, while the best random models used 600–800K.
- **The proxy ranking held up** at full training: the proxy-best model in each run was also the full-training best, or close to it.
- **Pipeline cost:** about 140 s of CPU LLM time per proposal. This is hidden by the asynchronous design (the LLM writes the next candidate while the GPU trains the current one).

**Next steps:** larger or API models (`llm.backend=openai`); show the parameter-budget headroom explicitly in the prompt; a novelty penalty or MAP-Elites archive to fight mode collapse; the `llm_noscore` and `zeroshot` ablations; 3 seeds per strategy.
