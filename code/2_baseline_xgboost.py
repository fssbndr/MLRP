import argparse
import os

import matplotlib.pyplot as plt
import polars as pl
from sklearn.metrics import auc, roc_curve

import xgboost as xgb
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

# Define output file paths
model_output_path = os.path.join(args.output_dir, "baseline_xgboost_model.json")
plot_output_path = os.path.join(args.plot_dir, "baseline_xgboost_roc_curve.png")

# Ensure output directories exist
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.plot_dir, exist_ok=True)

# ------------------------------------------------------------------------------
# load the processed data using the provided path
data = pl.read_parquet(args.input)

# filter out rows where the outcome variable is null
data = data.filter(pl.col("Mortality in ICU").is_not_null())

# Prepare features and target
X_all_df = data.select(
    "Pre-ICU LOS (days)",
    "Age (years)",
    "GCS",
    "HR",
    "MAP",
    "Urine output (ml)",
    "RR",
    "Temp (C)",
    "Admission Type",
    "Admission Urgency",
    "MechVent",
)
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
# XGBOOST MODEL
# Instantiate and fit the XGBoost Classifier on training data
model = xgb.XGBClassifier()
model.fit(X_train_np, y_train_np)

### SAVE MODEL ###
model.save_model(model_output_path)
################################################################################

### SAVE ROC CURVE ###
# Predict probabilities for the positive class on test data
y_pred_prob_test = model.predict_proba(X_test_np)[:, 1]

# Calculate AUC with bootstrapping
auc_stats = bootstrap_auc(
    y_test_np, y_pred_prob_test, n_bootstrap=100, random_state=42
)

# Calculate ROC curve using test data
fpr, tpr, thresholds = roc_curve(y_test_np, y_pred_prob_test)
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
plt.title("Receiver Operating Characteristic (ROC) Curve - XGBoost Test Set")
plt.legend(loc="lower right")

# Save the plot
plt.savefig(plot_output_path)
plt.close()
