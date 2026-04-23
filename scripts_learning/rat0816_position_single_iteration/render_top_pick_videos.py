import argparse
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RENDER_SCRIPT = SCRIPT_DIR / "render_perimeter_progress_paths.py"

TOP_PICKS = {
    "rat5_seed2": "/Users/Hannah/Programming/Hannahs-CEBRAs/outputs/rat0816_position_verbose_batch/rat5_position_verbose_rerun_2026-04-15_00-05-49_seed2",
    "rat5_seed1": "/Users/Hannah/Programming/Hannahs-CEBRAs/outputs/rat0816_position_verbose_batch/rat5_position_verbose_rerun_2026-04-14_23-58-58_seed1",
    "rat4_seed1": "/Users/Hannah/Programming/Hannahs-CEBRAs/outputs/rat0314_position_verbose_batch/rat4_position_verbose_rerun_2026-04-15_00-29-04_seed1",
    "rat4_seed3": "/Users/Hannah/Programming/Hannahs-CEBRAs/outputs/rat0314_position_verbose_batch/rat4_position_verbose_rerun_2026-04-15_00-49-28_seed3",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render perimeter-progress videos for the selected top position-decoding runs."
    )
    parser.add_argument("--bins", type=int, default=12)
    parser.add_argument("--band-fraction", type=float, default=0.30)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--elev", type=float, default=21.0)
    parser.add_argument("--azim", type=float, default=318.0)
    parser.add_argument(
        "--summary-path",
        type=str,
        default=str(SCRIPT_DIR / "top_pick_video_manifest.json"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary = {
        "renderer": str(RENDER_SCRIPT),
        "bins": args.bins,
        "band_fraction": args.band_fraction,
        "fps": args.fps,
        "dpi": args.dpi,
        "elev": args.elev,
        "azim": args.azim,
        "runs": [],
    }

    for label, run_dir_str in TOP_PICKS.items():
        run_dir = Path(run_dir_str).resolve()
        cmd = [
            sys.executable,
            str(RENDER_SCRIPT),
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
            "--elev",
            str(args.elev),
            "--azim",
            str(args.azim),
        ]
        subprocess.run(cmd, check=True)
        summary["runs"].append(
            {
                "label": label,
                "run_dir": str(run_dir),
                "video_path": str(run_dir / "perimeter_progress_paths.mp4"),
                "preview_path": str(run_dir / "perimeter_progress_paths_preview.png"),
                "static_svg": str(run_dir / "perimeter_progress_static.svg"),
                "static_png": str(run_dir / "perimeter_progress_static.png"),
                "summary_json": str(run_dir / "perimeter_progress_summary.json"),
            }
        )

    summary_path = Path(args.summary_path).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved top-pick video manifest to {summary_path}")


if __name__ == "__main__":
    main()
