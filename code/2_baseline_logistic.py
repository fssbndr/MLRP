import argparse
import os

import matplotlib.pyplot as plt
import polars as pl
from sklearn.metrics import auc, roc_curve

import statsmodels.formula.api as smf
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
    help="Path to the output directory for summary.",
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
summary_output_path = os.path.join(
    args.output_dir, f"baseline_logistic{suffix}_summary.txt"
)
plot_output_path = os.path.join(
    args.plot_dir, f"baseline_logistic{suffix}_roc_curve.png"
)

# ------------------------------------------------------------------------------
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
    # Stats dataset - use all statistical aggregations
    vital_features = [
        col
        for col in columns
        if any(stat in col for stat in ["mean", "std", "min", "max"])
    ]
elif has_minmax:
    # Minmax dataset - use only min/max aggregations
    vital_features = [
        col for col in columns if any(stat in col for stat in ["min", "max"])
    ]
elif has_hourly:
    # Hourly dataset - use all hourly columns
    vital_features = [col for col in columns if "Hour" in col]
else:
    # Regular dataset
    vital_features = ["GCS", "HR", "MAP", "RR", "Temp (C)"]

feature_columns = base_features + vital_features

# Prepare features and target
X_all_df = data.select([col for col in feature_columns if col in columns])
y_all_np = data.select("Mortality in ICU").to_numpy().flatten()

# Fill null values
categorical_cols = ["Admission Type", "Admission Urgency", "MechVent"]

# First, get all unique categorical values from the entire dataset to ensure consistency
all_categorical_values = {}
for col in categorical_cols:
    unique_vals = data.select(col).unique().to_numpy().flatten()
    # Remove None/null values and convert to strings
    unique_vals = [
        str(val)
        for val in unique_vals
        if val is not None and str(val) != "None"
    ]
    all_categorical_values[col] = unique_vals

X_all_df = X_all_df.with_columns(
    pl.when(pl.col(col).is_null())
    .then(pl.lit("Unknown"))
    .otherwise(pl.col(col))
    .cast(str)
    .alias(col)
    for col in categorical_cols
)
X_all_df = X_all_df.fill_null(0)

# Get train/test masks from the 'split_80_20' column
train_mask = (data["split_80_20"] == "train").to_numpy()
test_mask = (data["split_80_20"] == "test").to_numpy()

# Split features and target
X_train = X_all_df.filter(pl.Series(train_mask))
X_test = X_all_df.filter(pl.Series(test_mask))

y_train_np = y_all_np[train_mask]
y_test_np = y_all_np[test_mask]

# Prepare data for statsmodels.formula.api
feature_names = X_train.columns
target_name = "_target_"

X_train = X_train.with_columns(pl.Series(target_name, y_train_np).cast(int))

# Convert Polars DataFrames to Pandas DataFrames for statsmodels.formula.api
X_train_pd = X_train.to_pandas()
X_test_pd = X_test.to_pandas()

# Ensure categorical columns in test set only have levels seen in training set
for col in categorical_cols:
    train_levels = set(X_train_pd[col].unique())
    # Map any unseen levels in test set to 'Unknown'
    X_test_pd[col] = X_test_pd[col].apply(
        lambda x: x if x in train_levels else "Unknown"
    )
    # Ensure both train and test have the same categorical type
    X_train_pd[col] = X_train_pd[col].astype("category")
    X_test_pd[col] = X_test_pd[col].astype("category")

# Construct the formula string using Q() for quoting column names
formula_features_str = " + ".join(
    [
        (f'Q("{col}")' if not col in categorical_cols else f'C(Q("{col}"))')
        for col in feature_names
    ]
)
formula_str = f"{target_name} ~ {formula_features_str}"

# LOGISTIC REGRESSION MODEL using statsmodels.formula.api
# Fit the model on the training data
# sm.add_constant is not needed as smf handles the intercept by default
model = smf.logit(formula=formula_str, data=X_train_pd)
result = model.fit(disp=0, method="bfgs")

### SAVE SUMMARY ###
# Write summary to file using the constructed path (summary is from model trained on training data)
with open(summary_output_path, "w") as f:
    f.write(str(result.summary()))
################################################################################

### SAVE ROC CURVE ###
# Predict probabilities on the test set
y_pred_prob_test = result.predict(X_test_pd)

# Calculate AUC with bootstrapping
auc_stats = bootstrap_auc(
    y_test_np, y_pred_prob_test, n_bootstrap=100, random_state=42
)

# Calculate ROC curve using test data
fpr, tpr, thresholds = roc_curve(y_test_np, y_pred_prob_test)
roc_auc = auc(fpr, tpr)

# Plot ROC curve with confidence interval in title
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
plt.title("Receiver Operating Characteristic (ROC) Curve - Test Set")
plt.legend(loc="lower right")

# Save the plot
plt.savefig(plot_output_path)
plt.close()

### SAVE RESULTS TO CSV ###
# Create a DataFrame with patient IDs, actual labels, and predicted probabilities
test_icu_ids = data.filter(pl.Series(test_mask))["Global ICU Stay ID"].to_numpy()

predictions_df = pl.DataFrame(
    {
        "model_name": "sklearn.linear_model.LogisticRegression",
        "model_args": f"baseline_logistic{suffix}",
        "num_shots": 0,  # Baseline models don't use shots
        "auc_ci_lower": auc_stats["auc_ci_lower"],
        "auc_ci_upper": auc_stats["auc_ci_upper"],
        "global_icu_stay_id": test_icu_ids,
        "actual_label": y_test_np,
        "predicted_probability": y_pred_prob_test,
    }
)

# Save CSV results
csv_output_path = os.path.join(
    args.output_dir, f"baseline_logistic{suffix}_results.csv"
)
predictions_df.write_csv(csv_output_path)
