import argparse
import os

import matplotlib.pyplot as plt
import polars as pl
import numpy as np
from sklearn.metrics import auc, roc_curve
from lifelines import KaplanMeierFitter
from _utils import bootstrap_auc

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    required=True,
    help="Path to the input processed parquet data file.",
)
parser.add_argument(
    "--output_dir",
    required=True,
    help="Path to the output directory for results.",
)
parser.add_argument(
    "--plot_dir",
    required=True,
    help="Path to the output directory for the ROC curve plot.",
)
args = parser.parse_args()

# Ensure output directories exist
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.plot_dir, exist_ok=True)

# Define output file paths
summary_output_path = os.path.join(
    args.output_dir, "baseline_kaplan_meier_summary.txt"
)
plot_output_path = os.path.join(
    args.plot_dir, "baseline_kaplan_meier_roc_curve.png"
)
csv_output_path = os.path.join(
    args.output_dir, "baseline_kaplan_meier_results.csv"
)

# ------------------------------------------------------------------------------
# load the processed data using the provided path
data = pl.read_parquet(args.input)

# filter out rows where the outcome variable is null
data = data.filter(pl.col("Mortality in ICU").is_not_null())

# Get train/test masks from the 'split_80_20' column
train_mask = (data["split_80_20"] == "train").to_numpy()
test_mask = (data["split_80_20"] == "test").to_numpy()

# Split data
train_data = data.filter(pl.Series(train_mask))
test_data = data.filter(pl.Series(test_mask))

# Prepare survival data for training
train_durations = train_data["ICU LOS (days)"].to_numpy()
train_events = train_data["Mortality in ICU"].to_numpy().astype(bool)

# Censor observations at 3 days for prediction purposes
PREDICTION_HORIZON_DAYS = 3
train_durations_censored = np.minimum(train_durations, PREDICTION_HORIZON_DAYS)
train_events_censored = train_events & (
    train_durations <= PREDICTION_HORIZON_DAYS
)

################################################################################
# KAPLAN-MEIER MODEL
# Fit Kaplan-Meier estimator on training data
kmf = KaplanMeierFitter()
kmf.fit(train_durations_censored, train_events_censored, label="ICU Mortality")

# Calculate mortality probability for test set
test_durations = test_data["ICU LOS (days)"].to_numpy()
test_events = test_data["Mortality in ICU"].to_numpy()
test_icu_ids = test_data["Global ICU Stay ID"].to_numpy()

# Get the survival probability at prediction horizon from the fitted model
if PREDICTION_HORIZON_DAYS in kmf.survival_function_.index:
    survival_prob_horizon = kmf.survival_function_.loc[
        PREDICTION_HORIZON_DAYS
    ].iloc[0]
else:
    # Find closest time <= prediction horizon
    available_times = kmf.survival_function_.index
    valid_times = available_times[available_times <= PREDICTION_HORIZON_DAYS]
    if len(valid_times) > 0:
        closest_time = valid_times[-1]
        survival_prob_horizon = kmf.survival_function_.loc[closest_time].iloc[0]
    else:
        survival_prob_horizon = 1.0  # No events observed, assume survival

# Convert survival probability to mortality probability
mortality_prob_horizon = 1.0 - survival_prob_horizon

# For each test patient, predict mortality probability
y_pred_prob_test = []
y_test_np = []

for i, (duration, event) in enumerate(zip(test_durations, test_events)):
    # Actual label: True if patient died within prediction horizon
    if event and duration <= PREDICTION_HORIZON_DAYS:
        actual_mortality = 1
    else:
        actual_mortality = 0
    y_test_np.append(actual_mortality)

    # All patients get the same predicted probability based on population survival curve
    y_pred_prob_test.append(mortality_prob_horizon)

y_test_np = np.array(y_test_np)
y_pred_prob_test = np.array(y_pred_prob_test)

################################################################################

### SAVE SUMMARY ###
summary_stats = {
    "n_train": len(train_data),
    "n_test": len(test_data),
    "n_events_train": int(train_events_censored.sum()),
    "n_events_test_7day": int(y_test_np.sum()),
    "median_survival_time": kmf.median_survival_time_,
    "prediction_horizon_days": PREDICTION_HORIZON_DAYS,
}

with open(summary_output_path, "w") as f:
    f.write("Kaplan-Meier Survival Analysis Summary\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Training samples: {summary_stats['n_train']}\n")
    f.write(f"Test samples: {summary_stats['n_test']}\n")
    f.write(f"Training events (≤{PREDICTION_HORIZON_DAYS} days): {summary_stats['n_events_train']}\n")
    f.write(f"Test events (≤{PREDICTION_HORIZON_DAYS} days): {summary_stats['n_events_test_7day']}\n")
    f.write(f"Median survival time: {summary_stats['median_survival_time']:.2f} days\n")
    f.write(f"Prediction horizon: {PREDICTION_HORIZON_DAYS} days\n\n")

    # Add survival function summary
    f.write("Survival Function Summary:\n")
    f.write("-" * 30 + "\n")
    time_points = [1, 3, 7, 14, 30]
    for t in time_points:
        if t in kmf.survival_function_.index:
            surv_prob = kmf.survival_function_.loc[t].iloc[0]
            mort_prob = 1.0 - surv_prob
            f.write(f"Day {t:2d}: Survival = {surv_prob:.3f}, Mortality = {mort_prob:.3f}\n")

### CALCULATE METRICS ###
# Calculate AUC with bootstrapping
auc_stats = bootstrap_auc(
    y_test_np, y_pred_prob_test, n_bootstrap=100, random_state=42
)

# Calculate ROC curve using test data
fpr, tpr, thresholds = roc_curve(y_test_np, y_pred_prob_test)
roc_auc = auc(fpr, tpr)

### PLOT ROC CURVE ###
plt.figure(figsize=(8, 6))
plt.plot(
    fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})"
)
plt.plot(
    [0, 1],
    [0, 1],
    color="navy",
    lw=2,
    linestyle="--",
    label="Random classifier",
)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(
    f"Kaplan-Meier ROC Curve\n"
    f'AUC: {auc_stats["auc_mean"]:.3f} '
    f'(95% CI: {auc_stats["auc_ci_lower"]:.3f}-{auc_stats["auc_ci_upper"]:.3f})'
)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(plot_output_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"ROC curve plot saved to {plot_output_path}")

### SAVE RESULTS TO CSV ###
predictions_df = pl.DataFrame(
    {
        "model_name": "lifelines.KaplanMeierFitter",
        "model_args": "baseline_kaplan_meier",
        "num_shots": 0,  # Baseline models don't use shots
        "auc_ci_lower": auc_stats["auc_ci_lower"],
        "auc_ci_upper": auc_stats["auc_ci_upper"],
        "global_icu_stay_id": test_icu_ids,
        "actual_label": y_test_np,
        "predicted_probability": y_pred_prob_test,
    }
)

predictions_df.write_csv(csv_output_path)
print(f"Kaplan-Meier results saved to {csv_output_path}")
