import argparse
import os

import matplotlib.pyplot as plt
import polars as pl
from sklearn.metrics import auc, roc_curve
import lightgbm as lgb
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
    help="Path to the output directory for model.",
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

# ------------------------------------------------------------------------------
# load the processed data using the provided path
data = pl.read_parquet(args.input)

# filter out rows where the outcome variable is null
data = data.filter(pl.col("Mortality in ICU").is_not_null())

# Detect dataset type based on column names
columns = data.columns
has_hourly = any("Hour" in col for col in columns)
has_stats = any(
    "mean" in col or "std" in col or "min" in col or "max" in col
    for col in columns
)
has_minmax = any("min" in col or "max" in col for col in columns) and not any(
    "mean" in col or "std" in col for col in columns
)

# Check if this is hourly+stats dataset by looking at the input filename
is_hourly_stats_file = "hourly_stats" in args.input
is_minmax_file = "minmax" in args.input
is_forward_fill_file = "forward_fill" in args.input

# Determine suffix for output files based on dataset type
if is_hourly_stats_file or (has_hourly and has_stats):
    suffix = "_hourly_stats"
elif is_forward_fill_file:
    suffix = "_forward_fill"
elif is_minmax_file or has_minmax:
    suffix = "_minmax"
elif has_hourly:
    suffix = "_hourly"
elif has_stats:
    suffix = "_stats"
else:
    suffix = ""

# Define output file paths with dynamic naming
model_output_path = os.path.join(
    args.output_dir, f"baseline_lightgbm{suffix}_model.txt"
)
plot_output_path = os.path.join(
    args.plot_dir, f"baseline_lightgbm{suffix}_roc_curve.png"
)

# Prepare features based on dataset type
base_features = [
    "Pre-ICU LOS (days)",
    "Age (years)",
    "Urine output (ml)",
    "Admission Type",
    "Admission Urgency",
    "MechVent",
]

if has_stats:
    vital_features = [
        col
        for col in columns
        if any(stat in col for stat in ["mean", "std", "min", "max"])
    ]
elif has_minmax:
    vital_features = [
        col for col in columns if any(stat in col for stat in ["min", "max"])
    ]
elif has_hourly:
    vital_features = [col for col in columns if "Hour" in col]
else:
    vital_features = ["GCS", "HR", "MAP", "RR", "Temp (C)"]

feature_columns = base_features + vital_features

# Prepare features and target
X_all_df = data.select([col for col in feature_columns if col in columns])
y_all_np = data.select("Mortality in ICU").to_numpy().flatten()

# Fill null values
X_all_df = X_all_df.fill_null(0)

# Convert categorical columns to dummy variables
categorical_cols = ["Admission Type", "Admission Urgency"]
X_all_df_dummies = X_all_df.to_dummies(
    columns=categorical_cols, drop_first=True
)

# Get train/test masks from the 'split_80_20' column
train_mask = (data["split_80_20"] == "train").to_numpy()
test_mask = (data["split_80_20"] == "test").to_numpy()

# Split features and target
X_train_np = X_all_df_dummies.filter(pl.Series(train_mask)).to_numpy()
X_test_np = X_all_df_dummies.filter(pl.Series(test_mask)).to_numpy()

y_train_np = y_all_np[train_mask]
y_test_np = y_all_np[test_mask]

################################################################################
# LIGHTGBM MODEL
model = lgb.LGBMClassifier()
model.fit(X_train_np, y_train_np)
model.booster_.save_model(model_output_path)
################################################################################

### SAVE ROC CURVE ###
y_pred_prob_test = model.predict_proba(X_test_np)[:, 1]
auc_stats = bootstrap_auc(
    y_test_np, y_pred_prob_test, n_bootstrap=100, random_state=42
)
fpr, tpr, thresholds = roc_curve(y_test_np, y_pred_prob_test)
roc_auc = auc(fpr, tpr)

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
plt.title("Receiver Operating Characteristic (ROC) Curve - LightGBM Test Set")
plt.legend(loc="lower right")
plt.savefig(plot_output_path)
plt.close()

### SAVE RESULTS TO CSV ###
test_icu_ids = data.filter(pl.Series(test_mask))[
    "Global ICU Stay ID"
].to_numpy()
predictions_df = pl.DataFrame(
    {
        "model_name": "lightgbm.LGBMClassifier",
        "model_args": f"baseline_lightgbm{suffix}",
        "num_shots": 0,
        "auc_ci_lower": auc_stats["auc_ci_lower"],
        "auc_ci_upper": auc_stats["auc_ci_upper"],
        "global_icu_stay_id": test_icu_ids,
        "actual_label": y_test_np,
        "predicted_probability": y_pred_prob_test,
    }
)
csv_output_path = os.path.join(
    args.output_dir, f"baseline_lightgbm{suffix}_results.csv"
)
predictions_df.write_csv(csv_output_path)
