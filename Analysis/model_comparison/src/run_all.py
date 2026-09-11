"""Run the 9 model files (3 datasets x lazy / xgboost / nn) concurrently, one log per job.

Usage:  python run_all.py [--threads 2] [--datasets bus,route,combined]
Needs the Jupyter kernel `stinger-mc` (python -m ipykernel install --user --name stinger-mc).
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

MC = Path(__file__).resolve().parents[1]
LOGS = MC / "results" / "logs"


def jobs(datasets):
    nbconvert = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                 "--ExecutePreprocessor.kernel_name=stinger-mc", "--ExecutePreprocessor.timeout=-1"]
    for ds in datasets:
        folder = MC / "models" / ds
        yield f"lazy_{ds}", folder, nbconvert + [f"lazy_{ds}.ipynb"]
        yield f"xgboost_{ds}", folder, nbconvert + [f"xgboost_{ds}.ipynb"]
        yield f"nn_{ds}", folder, [sys.executable, f"nn_{ds}.py"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--datasets", default="bus,route,combined")
    args = ap.parse_args()
    n = str(args.threads)   # also cap OpenMP/BLAS pools (HistGB, LightGBM, XGBoost, numpy)
    env = {**os.environ, "MC_THREADS": n, "OMP_NUM_THREADS": n, "MKL_NUM_THREADS": n, "OPENBLAS_NUM_THREADS": n,
           "PYTHONIOENCODING": "utf-8"}
    env.pop("MC_SMOKE", None)                      # full budgets
    LOGS.mkdir(parents=True, exist_ok=True)
    running = {}
    for name, cwd, cmd in jobs(args.datasets.split(",")):
        log = open(LOGS / f"{name}.log", "w", encoding="utf-8")
        running[name] = (subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT), time.time(), log)
        print(f"started {name}", flush=True)
    failed = []
    for name, (proc, start, log) in running.items():
        code = proc.wait()
        log.close()
        print(f"{name}: exit {code} after {(time.time() - start) / 60:.1f} min", flush=True)
        if code:
            failed.append(name)
    print("FAILED: " + ", ".join(failed) if failed else "ALL OK", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
