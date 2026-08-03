# Running SCALA on vast.ai

A runbook for the 120B-token pretraining run, plus the failure modes that
actually bite on rented GPUs.

---

## 1. Pick an instance

```bash
pip install vastai && vastai set api-key <YOUR_KEY>

python vastai/plan_run.py --budget-tokens 120e9 --gpus 8            # offline estimate
python vastai/plan_run.py --search --gpu-name H200 --gpus 8         # live offers
python vastai/plan_run.py --search --create --gpu-name H200 --gpus 8
```

What to filter on, in priority order:

1. **`disk_space >= 1000` GB.** The tokenized 120B-token mixture is ~480 GB at
   uint32, plus the HF datasets cache during preparation. This is the single
   most common reason a run dies at hour 6.
2. **`inet_down >= 500` Mbps.** You are streaming several TB from the Hub during
   tokenization. At 100 Mbps that step alone takes days.
3. **`reliability > 0.98`** and **`verified=true`**. Interruptible instances are
   cheaper, and `resume: auto` makes them survivable — but only if you also set
   `save_every` low (1000 steps ≈ 33 minutes at 8×H100).
4. **NVLink / SXM over PCIe.** FSDP2 all-gathers a block's parameters every
   layer. The 4090/5090 rows in `plan_run.py` look cheap per FLOP, but without
   NVLink the all-gather dominates and real MFU lands well under the assumed
   35%. Consumer cards are fine for the tiny config, not for the 8B run.

## 2. Provision

```bash
vastai ssh <instance-id>
export REPO_URL=<your fork>            # or rsync the tree up
bash vastai/provision.sh
```

`provision.sh` installs system packages, creates a venv **with
`--system-site-packages`** so it keeps the image's driver-matched torch, points
every cache (`HF_HOME`, Triton, Inductor) at `/data` rather than the small
container root, and finishes by running `scripts/count_params.py` and the test
suite. If the tests fail, stop — do not start a paid training run.

## 3. Tokenize

```bash
tmux new -s prep
python scripts/prepare_data.py --config configs/data_ja_mix.yaml --dry-run   # check reachability first
python scripts/prepare_data.py --config configs/data_ja_mix.yaml \
    --out /data/tokens --budget-tokens 120e9 --workers 32
```

Notes:

* `--dry-run` pulls one record from every source and prints its keys. Run it
  before committing GPU-hours; dataset repos get renamed and gated regularly.
* Sources are independent. To parallelise across cheap CPU-only instances, run
  `--only <source>` on each and merge the `manifest.json` files.
* Re-running resumes: completed shards are detected and skipped.
* Gated repos need `huggingface-cli login` first.

## 4. Train

```bash
bash vastai/launch.sh                  # tmux, all GPUs, resume: auto
tmux attach -t photon
tail -f /workspace/runs/scala-8b-a1b/log.jsonl | jq -c '{step,ppl,grad_norm,"moe/maxvio_mean"}'
```

### What the first 200 steps should look like

| metric | healthy | trouble |
|---|---|---|
| `loss_token` | starts ≈ `ln(99584)` = 11.5, under 8 by step 500 | flat at 11.5 → check the loss shift (see `docs/architecture.md` §4) |
| `grad_norm` | 0.3–1.5, clipped occasionally | repeated spikes → lower `lr`, check `spike_guard` is on |
| `moe/maxvio_mean` | rises early, then falls below ~0.3 | stuck high → `bias_update_rate` too large, see §6 |
| `loss_rec` | falls, plateaus well above 0 | never plateaus to 0 — that is expected, it is a prediction objective |
| `tok_per_s` | ~1.1 M on 8×H100 | much lower → see §5 |

### Batch-size warmup

`global_batch_tokens` is 2 M, ramped from 25% over the first 8 B tokens. The
trainer implements this by varying gradient accumulation, so `micro_batch_size`
stays constant and memory does not move.

## 5. Throughput troubleshooting

* **First 100 steps are slow.** `torch.compile` is warming up. The Inductor
  cache lives on `/data` so a restart does not repay it.
* **`torch._grouped_mm unavailable` warning.** The MoE fell back to a per-expert
  loop — correct but slow, and it syncs to host once per layer. Needs
  torch ≥ 2.9 on CUDA with bf16 inputs. Check with
  `python -c "import torch; print(hasattr(torch,'_grouped_mm'))"`.
* **Recompilation storms.** MoE dispatch shapes are data-dependent. The trainer
  deliberately compiles `ffn.experts` with `dynamic=True` and leaves routing
  eager. If you compile the whole block, expect Dynamo to recompile forever.
* **MFU below 20%.** Usually the dataloader. Raise `num_workers`, and confirm
  `/data` is a real NVMe volume, not network storage.

## 6. Stability

* `moe_bias_freeze_frac: 0.95` sets γ → 0 for the final 5%, as DeepSeek-V3 does.
* γ itself must stay small relative to the sigmoid score scale. Measured on the
  tiny config: γ=0.002 drives MaxVio 0.096 → 0.037; γ=0.05 oscillates and stalls
  at 0.265. The default 1e-3 is correct — do not "help" it by raising it.
* QK-Norm is on everywhere, which is why there is no QK-Clip
  (see `docs/architecture.md` §9).
* `GradNormGuard` skips steps whose grad norm exceeds 4× the running median,
  capped at 2% of steps so it can never silently mask a genuine divergence.

## 7. Checkpoints and getting the model off the box

Rented instances disappear. Push checkpoints off the machine.

```bash
# consolidate a distributed checkpoint into one safetensors file + model card
python scripts/export_checkpoint.py /workspace/runs/scala-8b-a1b/final \
    --out /data/export/scala-8b-a1b \
    --push-to <hf-user>/scala-8b-a1b
```

For periodic sync during the run, `vastai/sync.sh` mirrors the newest checkpoint
to the Hub and prunes old local ones.

## 8. Evaluate

```bash
python scripts/eval_ja.py --ckpt /workspace/runs/scala-8b-a1b/final \
    --tasks jcommonsenseqa jnli marc_ja jmmlu --limit 500

python scripts/benchmark_generation.py --config configs/base_8b_a1b.yaml \
    --ckpt /workspace/runs/scala-8b-a1b/final \
    --prompt-len 4096 --new-tokens 512 --check-agreement
```

The `--check-agreement` number (RecGen vs HierGen greedy match) is the gate for
shipping RecGen. If it is low, `rec_loss_alpha` and/or `self_cond_prob_end` were
too small — both are cheap to raise in a short continued-training phase, because
only the decoders are affected.

For the full llm-jp-eval / Nejumi suite you need a `transformers`-compatible
wrapper; PHOTON is not an upstream architecture. `scala/eval/harness.py`
implements the same log-likelihood multiple-choice protocol natively, which is
enough to track progress during pretraining.

## 9. Cost reality check

At MFU 35%, 120B tokens on 8×H200 is ~1.3 days and roughly $640. Budget
**about 1.5×** that: tokenization time, a failed instance or two, and the first
run always teaching you something that justifies a restart.

Cut cost by lowering `total_tokens` first — the WSD schedule's stable phase
means you can stop early and only need to re-plan the cooldown.
