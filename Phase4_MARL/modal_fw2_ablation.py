"""Modal cloud launcher for the FW2 ablation (SAC/PPO x curriculum on S_AIR_M).

Runs the 2x2 attribution ablation on the airside arm in the cloud, 5 seeds each:

    SAC_curr, SAC_nocurr, PPO_curr, PPO_nocurr

It REUSES the local FW2 code (fw2_sac.train_sac_curriculum) inside the container,
so there is a single source of truth for the env, the curriculum and the eval.

Resilience: every job writes its result JSON to the shared Volume and commits, so
a dropped local connection never loses completed work. Launch detached and collect
later:

    modal run --detach modal_fw2_ablation.py --seeds 5 --timesteps 200000
    modal run modal_fw2_ablation.py --collect      # gather whatever finished

Local paths are resolved absolutely from this file, so it runs from any cwd. The
FW2 target S_AIR_M needs the Phase 1 airside annual CSV (mounted below) on top of
the Phase 2 CSV that modal_train_phase4.py already uses.

Author: Fede - Master's thesis, Politecnico di Torino, 2026.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import List, Tuple

import modal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("fw2.modal")

APP_NAME = "phase4-fw2-ablation"
VOLUME_NAME = "phase4-models"
VOLUME_MOUNT = "/models"
REMOTE_CODE_DIR = "/root/Phase4"
RESULTS_SUBDIR = "fw2_results"

PHASE2_CSV_REL = "Phase2_ISMatch/results/step_2_4_delta_tc_calibration_lc.csv"
AIRSIDE_CSV_REL = "Phase1_PhysicalDT/airside/results/datacenter_dt_results_annual.csv"
REMOTE_PHASE2_CSV = f"/root/{PHASE2_CSV_REL}"
REMOTE_AIRSIDE_CSV = f"/root/{AIRSIDE_CSV_REL}"

LOCAL_CODE_DIR = Path(__file__).resolve().parent
LOCAL_PHASE2_CSV = LOCAL_CODE_DIR.parent / PHASE2_CSV_REL
LOCAL_AIRSIDE_CSV = LOCAL_CODE_DIR.parent / AIRSIDE_CSV_REL

CELLS: Tuple[Tuple[str, bool], ...] = (("sac", True), ("sac", False), ("ppo", True), ("ppo", False))
DEFAULT_SEEDS = 5
DEFAULT_TIMESTEPS = 200_000

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.11.0+cpu", extra_index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("stable-baselines3==2.8.0", "gymnasium==1.2.2", "numpy==2.4.4",
                 "pandas>=2.2", "scipy>=1.10", "tensorboard>=2.16")
    .env({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    .add_local_file(str(LOCAL_PHASE2_CSV), remote_path=REMOTE_PHASE2_CSV)
    .add_local_file(str(LOCAL_AIRSIDE_CSV), remote_path=REMOTE_AIRSIDE_CSV)
    .add_local_dir(str(LOCAL_CODE_DIR), remote_path=REMOTE_CODE_DIR,
                   ignore=["**/__pycache__", "**/*.zip", "**/results/**", "**/logs/**",
                           "**/runs/**", "**/models/**"])
)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(image=image, volumes={VOLUME_MOUNT: volume}, cpu=8.0,
              timeout=2 * 60 * 60, retries=2)
def train_one(algo: str, curriculum: bool, seed: int, total_timesteps: int) -> dict:
    """Train one ablation cell on S_AIR_M, persist the result to the Volume, return it."""
    import json
    import sys

    if REMOTE_CODE_DIR not in sys.path:
        sys.path.insert(0, REMOTE_CODE_DIR)

    import torch

    from fw2_sac.train_sac_curriculum import Config, build_world, train_one_seed

    torch.set_num_threads(1)
    cfg = Config(algo=algo, curriculum=curriculum, timesteps=total_timesteps, n_seeds=1, smoke=False)
    cfg.models_dir = Path(VOLUME_MOUNT) / "fw2_models"
    cfg.runs_dir = Path(VOLUME_MOUNT) / "fw2_runs"
    cfg.out_dir = Path("/tmp/fw2_out")
    for d in (cfg.models_dir, cfg.runs_dir, cfg.out_dir):
        d.mkdir(parents=True, exist_ok=True)

    world = build_world(cfg)
    result = train_one_seed(cfg, world, seed, algo, curriculum)

    res_dir = Path(VOLUME_MOUNT) / RESULTS_SUBDIR
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / f"{result['config']}_seed{seed}.json").write_text(json.dumps(result))
    volume.commit()
    logger.info("done %s seed=%d conv=%.3f poa=%.4f w_dc=%.0f (persisted to Volume)",
                result["config"], seed, result["convergence_rate"], result["final_poa"],
                result["welfare_dc"])
    return result


@app.function(image=image, volumes={VOLUME_MOUNT: volume})
def collect_results() -> List[dict]:
    """Read every persisted per-job result JSON from the Volume."""
    import json

    volume.reload()
    res_dir = Path(VOLUME_MOUNT) / RESULTS_SUBDIR
    rows: List[dict] = []
    if res_dir.exists():
        for p in sorted(res_dir.glob("*.json")):
            try:
                rows.append(json.loads(p.read_text()))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("skip %s: %s", p.name, exc)
    return rows


def _build_jobs(mode: str, seeds: int, timesteps: int) -> List[Tuple[str, bool, int, int]]:
    if mode == "all":
        cells = list(CELLS)
    else:
        label_map = {"sac_curr": ("sac", True), "sac_nocurr": ("sac", False),
                     "ppo_curr": ("ppo", True), "ppo_nocurr": ("ppo", False)}
        if mode not in label_map:
            raise ValueError(f"Unknown mode {mode!r}; expected all or one of {list(label_map)}.")
        cells = [label_map[mode]]
    return [(algo, curr, seed, timesteps) for (algo, curr) in cells for seed in range(seeds)]


def _write_results(rows: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["config", "scenario", "seed", "final_poa", "welfare_dc", "welfare_mfg",
              "shapley_fairness", "n_rounds", "convergence_rate"]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["config"], r["seed"])):
            w.writerow({k: r.get(k) for k in fields})


def _summary(rows: List[dict]) -> None:
    import statistics

    cells: dict = {}
    for r in rows:
        cells.setdefault(r["config"], []).append(r)
    print(f"{'cell':>11} {'n':>3} | {'conv_rate':>9} {'PoA':>7} {'w_dc':>12}")
    print("-" * 50)
    for label, recs in sorted(cells.items()):
        conv = statistics.fmean(r["convergence_rate"] for r in recs)
        poas = [r["final_poa"] for r in recs if r["final_poa"] == r["final_poa"]]
        poa = statistics.fmean(poas) if poas else float("nan")
        w_dc = statistics.fmean(r["welfare_dc"] for r in recs)
        print(f"{label:>11} {len(recs):>3} | {conv:>9.3f} {poa:>7.3f} {w_dc:>12,.0f}")
    print("\nCanonical S_AIR_M shielded baseline: D conv_rate 0.41, CI95 [0.255, 0.558], n=10 (Tab 5.8).")


@app.local_entrypoint()
def main(mode: str = "all", seeds: int = DEFAULT_SEEDS, timesteps: int = DEFAULT_TIMESTEPS,
         collect: bool = False) -> None:
    """Dispatch the ablation, or (with --collect) gather persisted results from the Volume."""
    if collect:
        rows = collect_results.remote()
        out_path = LOCAL_CODE_DIR / "results" / f"modal_fw2_collected_{len(rows)}rows.csv"
        _write_results(rows, out_path)
        logger.info("Collected %d results to %s", len(rows), out_path)
        _summary(rows)
        return

    jobs = _build_jobs(mode=mode, seeds=seeds, timesteps=timesteps)
    logger.info("Dispatching %d FW2 jobs (mode=%s, seeds=%d, timesteps=%d)",
                len(jobs), mode, seeds, timesteps)
    rows: List[dict] = list(train_one.starmap(jobs))
    out_path = LOCAL_CODE_DIR / "results" / f"modal_fw2_{mode}_{seeds}seed_{timesteps // 1000}k.csv"
    _write_results(rows, out_path)
    logger.info("Wrote %d results to %s", len(rows), out_path)
    _summary(rows)
