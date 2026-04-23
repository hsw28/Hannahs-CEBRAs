import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON = Path("/Users/Hannah/anaconda3/envs/CEBRA/bin/python")
RUNNER = SCRIPT_DIR / "run_position_single_iteration_generic.py"

ENV = os.environ.copy()
ENV.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
ENV["OMP_NUM_THREADS"] = "1"
ENV["MKL_NUM_THREADS"] = "1"
ENV["OPENBLAS_NUM_THREADS"] = "1"
ENV["NUMEXPR_NUM_THREADS"] = "1"


def run_job(job, log_handle):
    command = [
        str(PYTHON),
        str(RUNNER),
        job["traceA1An_An"],
        job["traceAnB1_An"],
        job["traceA1An_A1"],
        job["traceAnB1_B1"],
        job["PosAn"],
        job["PosA1"],
        job["PosB1"],
        "--run-label",
        job["run_label"],
        "--output-dir",
        job["output_dir"],
        "--learning-rate",
        str(job["learning_rate"]),
        "--max-iterations",
        str(job["max_iterations"]),
        "--distance",
        job["distance"],
        "--temp-mode",
        job["temp_mode"],
        "--seed",
        str(job["seed"]),
    ]
    if job["min_temperature"] is not None:
        command.extend(["--min-temperature", str(job["min_temperature"])])

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_handle.write(f"\n[{timestamp}] START {job['run_label']} seed={job['seed']}\n")
    log_handle.write("COMMAND: " + " ".join(command) + "\n")
    log_handle.flush()

    completed = subprocess.run(
        command,
        env=ENV,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_handle.write(f"\n[{timestamp}] END {job['run_label']} seed={job['seed']} rc={completed.returncode}\n")
    log_handle.flush()
    return completed.returncode


def main():
    manifest_path = Path(sys.argv[1]).resolve()
    manifest = json.loads(manifest_path.read_text())
    log_path = Path(manifest["log_path"]).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    failures = []
    with open(log_path, "a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n=== Batch started {datetime.now().isoformat()} ===\n")
        for job in manifest["jobs"]:
            rc = run_job(job, log_handle)
            if rc != 0:
                failures.append(
                    {
                        "run_label": job["run_label"],
                        "seed": job["seed"],
                        "returncode": rc,
                    }
                )
        log_handle.write(f"=== Batch finished {datetime.now().isoformat()} ===\n")

    summary = {
        "manifest": str(manifest_path),
        "log_path": str(log_path),
        "n_jobs": len(manifest["jobs"]),
        "failures": failures,
        "completed_at": datetime.now().isoformat(),
    }
    summary_path = Path(manifest["summary_path"]).resolve()
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
