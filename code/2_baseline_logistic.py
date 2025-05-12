import argparse
import os

import matplotlib.pyplot as plt
import polars as pl
from sklearn.metrics import auc, roc_curve

import statsmodels.api as sm

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
    help="Path to the output directory for summary.",
)
parser.add_argument(
    "--plot_dir",
    required=True,
    help="Path to the output directory for the ROC curve plot.",
)
args = parser.parse_args()

# Define output file paths
summary_output_path = os.path.join(
    args.output_dir, "baseline_logistic_summary.txt"
)
plot_output_path = os.path.join(
    args.plot_dir, "baseline_logistic_roc_curve.png"
)

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
# Convert to numpy and add constant for train and test sets
X_train_final_np = sm.add_constant(X_train_np)
X_test_final_np = sm.add_constant(X_test_np)

# LOGISTIC REGRESSION MODEL
# Fit the model on the training data
model = sm.Logit(y_train_np, X_train_final_np)
result = model.fit(disp=0)

### SAVE SUMMARY ###
# Write summary to file using the constructed path (summary is from model trained on training data)
with open(summary_output_path, "w") as f:
    f.write(str(result.summary()))
################################################################################

### SAVE ROC CURVE ###
# Predict probabilities on the test set
y_pred_prob_test = result.predict(X_test_final_np)

# Calculate ROC curve using test data
fpr, tpr, thresholds = roc_curve(y_test_np, y_pred_prob_test)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.2f}) on Test Set")
plt.plot([0, 1], [0, 1], color="black", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve - Test Set")
plt.legend(loc="lower right")

# Save the plot
plt.savefig(plot_output_path)
plt.close()
