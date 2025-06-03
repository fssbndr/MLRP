import argparse
import os

import matplotlib.pyplot as plt
import polars as pl
from sklearn.metrics import auc, roc_curve
from _utils import bootstrap_auc

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input", required=True, help="Path to the input parquet data file."
)
parser.add_argument(
    "--plot_dir",
    required=True,
    help="Path to the output directory for the ROC curve plot.",
)
args = parser.parse_args()

# Define output plot path
plot_output_path = os.path.join(args.plot_dir, "baseline_oasis_roc_curve.png")

# Ensure plot output directory exists
os.makedirs(args.plot_dir, exist_ok=True)

# load the data using the provided path
data = (
    pl.read_parquet(args.input)
    .group_by("Global ICU Stay ID")
    .agg(
        pl.col("Mortality in ICU").first(),
        pl.col("OASIS ICU Mortality Rate").first(),
    )
)

### SAVE ROC CURVE ###
# Filter out rows where the outcome variable or prediction is null
data = data.filter(pl.col("Mortality in ICU").is_not_null())
data = data.filter(pl.col("OASIS ICU Mortality Rate").is_not_null())

# Define y (true outcome) and y_pred_prob (OASIS score)
y = data.select("Mortality in ICU").to_numpy().flatten()
y_pred_prob = data.select("OASIS ICU Mortality Rate").to_numpy().flatten()

# Calculate AUC with bootstrapping
auc_stats = bootstrap_auc(y, y_pred_prob, n_bootstrap=100, random_state=42)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve with confidence interval
plt.figure()
plt.plot(
    fpr,
    tpr,
    lw=2,
    label=f"ROC curve (AUC = {auc_stats['auc']:.3f} [{auc_stats['auc_ci_lower']:.3f}-{auc_stats['auc_ci_upper']:.3f}])",
)
plt.plot([0, 1], [0, 1], color="black", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve - OASIS Baseline")
plt.legend(loc="lower right")

# Save the plot
plt.savefig(plot_output_path)  # Use defined path
plt.close()
