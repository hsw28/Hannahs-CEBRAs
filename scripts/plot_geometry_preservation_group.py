import argparse
import os
import sys
from datetime import datetime

sys.path.append('/Users/Hannah/Programming/Hannahs-CEBRAs/')
sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs')

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cond_geometry_preservation import paired_geometry_stats


parser = argparse.ArgumentParser(description="Plot CEBRA geometry-preservation group summary from per-rat summary CSVs.")
parser.add_argument("summary_csvs", nargs="+", help="Per-rat *_summary.csv files from cond_geometry_preservation_script.py.")
parser.add_argument("--output_dir", default="geometry_preservation_outputs", help="Directory for the group plot and CSV.")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

frames = []
for path in args.summary_csvs:
    frame = pd.read_csv(path)
    if "rat_id" not in frame or frame["rat_id"].isna().all():
        frame["rat_id"] = os.path.basename(path).split("_summary.csv")[0]
    frames.append(frame)

data = pd.concat(frames, ignore_index=True)
if "rReal" not in data.columns and "real_score" in data.columns:
    data["rReal"] = data["real_score"]
if "rShuff" not in data.columns:
    if "shuffle_score" in data.columns:
        data["rShuff"] = data["shuffle_score"]
    elif "shuffle_mean" in data.columns:
        data["rShuff"] = data["shuffle_mean"]

rat_summary = (
    data.groupby("rat_id", dropna=False)
    .agg(real_mean=("rReal", "mean"), shuff_mean=("rShuff", "mean"))
    .reset_index()
)

stats = paired_geometry_stats(rat_summary["real_mean"], rat_summary["shuff_mean"])
group_summary = pd.DataFrame([{**stats}])

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
summary_path = os.path.join(args.output_dir, f"geometry_preservation_group_summary_{timestamp}.csv")
plot_path = os.path.join(args.output_dir, f"geometry_preservation_group_{timestamp}.png")
group_summary.to_csv(summary_path, index=False)

fig, ax = plt.subplots(figsize=(4.8, 4.6))
means = [stats["shuff_mean"], stats["real_mean"]]
errors = [stats["shuff_sem"], stats["real_sem"]]
ax.bar([0, 1], means, yerr=errors, color=["#d8dadd", "#c2410c"], edgecolor="#333333", capsize=4, width=0.6)
ax.scatter(np.zeros(len(rat_summary)) - 0.08, rat_summary["shuff_mean"], color="#6b7280", s=35, alpha=0.8)
ax.scatter(np.ones(len(rat_summary)) + 0.08, rat_summary["real_mean"], color="#7c2d12", s=35, alpha=0.9)
for _, row in rat_summary.iterrows():
    ax.plot([0 - 0.08, 1 + 0.08], [row["shuff_mean"], row["real_mean"]], color="#9ca3af", linewidth=0.8, alpha=0.7)

ax.set_xticks([0, 1])
ax.set_xticklabels(["Shuffled", "Real"])
ax.set_ylabel("Mean Spearman geometry-preservation score")
ax.set_title("Group Geometry Preservation")
ax.axhline(0, color="#9ca3af", linewidth=0.8)
fig.tight_layout()
fig.savefig(plot_path, dpi=300)
plt.close(fig)

print(f"Group summary saved to {summary_path}")
print(f"Group plot saved to {plot_path}")
