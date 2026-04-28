import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


PYTHON = Path("/Users/Hannah/anaconda3/envs/CEBRA/bin/python")
RENDERER = Path("/Users/Hannah/Programming/Hannahs-CEBRAs/scripts_learning/rat0816_position_single_iteration/render_perimeter_progress_paths.py")
RUN_ENV = {
    **os.environ,
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "MPLCONFIGDIR": "/tmp/matplotlib",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Watch output roots and render perimeter-progress videos for newly completed runs."
    )
    parser.add_argument("--root", action="append", required=True, help="Output root to watch.")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--bins", type=int, default=12)
    parser.add_argument("--band-fraction", type=float, default=0.30)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")


def run_dirs(root: Path):
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if "position_verbose_rerun_corrected_" not in path.name:
            continue
        yield path


def is_complete(run_dir: Path) -> bool:
    return (run_dir / "run_summary.json").exists() and (run_dir / "model_b1_matched.pt").exists()


def is_rendered(run_dir: Path) -> bool:
    return (run_dir / "perimeter_progress_summary.json").exists()


def render_run(run_dir: Path, args, log_path: Path, jsonl_path: Path) -> None:
    cmd = [
        str(PYTHON),
        str(RENDERER),
        "--run-dir",
        str(run_dir),
        "--bins",
        str(args.bins),
        "--band-fraction",
        str(args.band_fraction),
        "--fps",
        str(args.fps),
        "--dpi",
        str(args.dpi),
    ]
    render_log = run_dir / "auto_render.log"
    started = time.time()
    print(f"[{now()}] rendering {run_dir}", flush=True)
    with render_log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
            env=RUN_ENV,
        )
    elapsed = round(time.time() - started, 2)
    summary_path = run_dir / "perimeter_progress_summary.json"
    if proc.returncode == 0 and summary_path.exists():
        message = f"[{now()}] render complete for {run_dir.name} in {elapsed:.2f}s"
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(message + "\n")
        append_jsonl(
            jsonl_path,
            {
                "timestamp": datetime.now().isoformat(),
                "run_dir": str(run_dir),
                "rendered": True,
                "elapsed_seconds": elapsed,
                "summary_path": str(summary_path),
                "video_path": str(run_dir / "perimeter_progress_paths.mp4"),
            },
        )
        return

    message = f"[{now()}] render failed for {run_dir.name} with exit code {proc.returncode}"
    print(message, flush=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(message + "\n")
    append_jsonl(
        jsonl_path,
        {
            "timestamp": datetime.now().isoformat(),
            "run_dir": str(run_dir),
            "rendered": False,
            "elapsed_seconds": elapsed,
            "returncode": proc.returncode,
            "render_log": str(render_log),
        },
    )


def main():
    args = parse_args()
    roots = [Path(root).resolve() for root in args.root]
    seen_complete = set()
    for root in roots:
        for run_dir in run_dirs(root):
            if is_complete(run_dir):
                seen_complete.add(str(run_dir))

    print(f"[{now()}] watching {len(roots)} roots for newly completed runs", flush=True)
    while True:
        for root in roots:
            log_path = root / "auto_render_watcher.log"
            jsonl_path = root / "auto_render_watcher_events.jsonl"
            for run_dir in run_dirs(root):
                run_key = str(run_dir)
                complete = is_complete(run_dir)
                if not complete:
                    continue
                if run_key not in seen_complete:
                    seen_complete.add(run_key)
                    if is_rendered(run_dir):
                        message = f"[{now()}] already rendered: {run_dir.name}"
                        print(message, flush=True)
                        with log_path.open("a", encoding="utf-8") as fh:
                            fh.write(message + "\n")
                        append_jsonl(
                            jsonl_path,
                            {
                                "timestamp": datetime.now().isoformat(),
                                "run_dir": run_key,
                                "rendered": True,
                                "reason": "already rendered",
                            },
                        )
                    else:
                        render_run(run_dir, args, log_path, jsonl_path)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
