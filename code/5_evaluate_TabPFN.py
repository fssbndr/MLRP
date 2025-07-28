import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
    accuracy_score,
)
from tabpfn import TabPFNRegressor
from _utils import bootstrap_auc

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Set device preference: CUDA > MPS > CPU
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Set device for TabPFN
torch.set_default_device(device)
print(f"Using device: {device}")

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Evaluate TabPFN with varying few-shot examples."
)
parser.add_argument(
    "--processed_data_path",
    type=str,
    required=True,
    help="Path to the processed_data.parquet file.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Directory to save the evaluation results CSV file.",
)
parser.add_argument(
    "--plot_dir",
    type=str,
    required=True,
    help="Directory to save the ROC curve plot.",
)
parser.add_argument(
    "--num_shots",
    type=int,
    default=0,
    help="Number of examples per class for few-shot training (0 for full training set).",
)
parser.add_argument(
    "--hourly",
    action="store_true",
    help="Use hourly aggregated data (changes model name suffix for output files).",
)
parser.add_argument(
    "--stats",
    action="store_true",
    help="Use comprehensive statistics aggregated data (changes model name suffix for output files).",
)
parser.add_argument(
    "--minmax",
    action="store_true",
    help="Use min/max statistics aggregated data (changes model name suffix for output files).",
)
parser.add_argument(
    "--forward-fill",
    action="store_true",
    help="Use forward-fill hourly data (changes model name suffix for output files).",
)
args = parser.parse_args()

# Define output file paths
model_suffix = "_basic"
if args.hourly:
    model_suffix = "_hourly"
if args.forward_fill:
    model_suffix = "_forward_fill"
if args.stats:
    model_suffix = "_stats"
if args.hourly and args.stats:
    model_suffix = "_hourly_stats"
if args.minmax:
    model_suffix = "_minmax"
model_name = f"tabpfn{model_suffix}"

output_csv_path = os.path.join(
    args.output_dir, f"{model_name}_{args.num_shots}-shot_results.csv"
)
plot_output_path = os.path.join(
    args.plot_dir, f"{model_name}_{args.num_shots}-shot_roc_curve.png"
)

# Ensure output directories exist
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.plot_dir, exist_ok=True)

# ------------------------------------------------------------------------------
# load the processed data using the provided path
data = pl.read_parquet(args.processed_data_path)

# filter out rows where the outcome variable is null
data = data.filter(pl.col("Mortality in ICU").is_not_null())

# Prepare features and target
X_all_df = data.select(
    "Pre-ICU LOS (days)",
    "Age (years)",
    pl.col("^GCS.*$"),
    pl.col("^HR.*$"),
    pl.col("^MAP.*$"),
    "Urine output (ml)",
    pl.col("^RR.*$"),
    pl.col("^Temp (C).*$"),
    "Admission Type",
    "Admission Urgency",
    "MechVent",
)
y_all_np = data.select("Mortality in ICU").to_numpy().flatten()
icu_ids_all_series = data["Global ICU Stay ID"]

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

# Get ICU IDs for the test set
icu_ids_test_series = icu_ids_all_series.filter(pl.Series(test_mask))
icu_ids_test_np = icu_ids_test_series.to_numpy().flatten()

# Prepare Polars DataFrames for few-shot sampling from the training set
X_train_full_df = X_all_df_dummies.filter(pl.Series(train_mask))
y_train_full_series = (
    data.filter(pl.Series(train_mask)).select("Mortality in ICU").to_series()
)


# Prepare training data based on num_shots
if args.num_shots <= 0:
    print("Error: --num_shots must be greater than 0 for few-shot evaluation.")
    exit(0)  # exit without error

# Few-shot examples preparation
print(f"Preparing {args.num_shots}-shot examples for TabPFN training...")
train_df = X_train_full_df.with_columns(
    y_train_full_series.alias("mortality_in_icu")
)

died_df = train_df.filter(pl.col("mortality_in_icu") == 1)
survived_df = train_df.filter(pl.col("mortality_in_icu") == 0)

n_sample_died = min(args.num_shots, len(died_df))
n_sample_survived = min(args.num_shots, len(survived_df))

# Print warnings if fewer examples are available than requested
if n_sample_died < args.num_shots:
    print(
        f"Warning: Only {n_sample_died} examples available for 'died' class. "
        f"Using {n_sample_died} instead of {args.num_shots}."
    )
if n_sample_survived < args.num_shots:
    print(
        f"Warning: Only {n_sample_survived} examples available for 'survived' class. "
        f"Using {n_sample_survived} instead of {args.num_shots}."
    )

few_shot_list = []

# Sample from 'died' class
if n_sample_died > 0:
    sampled_died = died_df.sample(n=n_sample_died, seed=SEED, shuffle=True)
    few_shot_list.append(sampled_died)

# Sample from 'survived' class
if n_sample_survived > 0:
    sampled_survived = survived_df.sample(
        n=n_sample_survived, seed=SEED, shuffle=True
    )
    few_shot_list.append(sampled_survived)

# Concatenate the sampled dataframes and shuffle to mix died/survived cases
few_shot_examples = pl.concat(few_shot_list).sample(
    fraction=1.0, shuffle=True, seed=SEED
)

X_fit_df = few_shot_examples.drop("mortality_in_icu")
y_fit_series = few_shot_examples["mortality_in_icu"]

X_fit_np = X_fit_df.to_numpy()
y_fit_np = y_fit_series.to_numpy().flatten()

print(
    f"Training TabPFN on {len(X_fit_np)} few-shot samples ({n_sample_died} died, {n_sample_survived} survived)."
)

################################################################################
# TABPFN MODEL
# Instantiate and fit the TabPFN Regressor on training data
regressor = TabPFNRegressor()
regressor.fit(X_fit_np, y_fit_np)
y_pred_prob_test = regressor.predict(X_test_np)

actual_labels = y_test_np
predictions = (y_pred_prob_test > 0.5).astype(int)
predicted_probabilities = y_pred_prob_test
################################################################################

### CALCULATE METRICS ###

# Calculate AUC with bootstrapping
auc_stats = bootstrap_auc(
    actual_labels, predicted_probabilities, n_bootstrap=100, random_state=SEED
)

model_metrics = {
    "accuracy": accuracy_score(actual_labels, predictions),
    "auc": auc_stats["auc"],
    "auc_ci_lower": auc_stats["auc_ci_lower"],
    "auc_ci_upper": auc_stats["auc_ci_upper"],
    "f1": f1_score(actual_labels, predictions, zero_division=0),
    "cm": confusion_matrix(actual_labels, predictions).tolist(),
}
print(
    f"Model: {model_name}, "
    f"Num Shots: {args.num_shots}, "
    f"Accuracy: {model_metrics['accuracy']:.2f}, "
    f"AUC: {model_metrics['auc']:.3f} [{model_metrics['auc_ci_lower']:.3f}-{model_metrics['auc_ci_upper']:.3f}], "
    f"F1 Score: {model_metrics['f1']:.2f}, "
    f"Confusion Matrix: {model_metrics['cm']}"
)

### SAVE RESULTS ###
# Create a Polars DataFrame with patient IDs, actual labels, and predictions
predictions_df = pl.DataFrame(
    {
        "model_name": "PriorLabs/TabPFNv2",
        "model_args": model_name,
        "num_shots": args.num_shots,
        "auc_ci_lower": auc_stats["auc_ci_lower"],
        "auc_ci_upper": auc_stats["auc_ci_upper"],
        "global_icu_stay_id": icu_ids_test_np,
        "actual_label": actual_labels,
        "predicted_probability": predicted_probabilities,
    }
)

predictions_df.write_csv(output_csv_path)
print(f"Per-patient evaluation results for {model_name} saved to {output_csv_path}") # fmt: skip

### SAVE ROC CURVE ###
# Calculate ROC curve using actual labels and predicted probabilities
fpr, tpr, thresholds = roc_curve(actual_labels, predicted_probabilities)
roc_auc_val = auc(fpr, tpr)

plt.figure()
plt.plot(
    fpr,
    tpr,
    lw=2,
    label=f"ROC curve (AUC = {roc_auc_val:.2f} [{auc_stats['auc_ci_lower']:.2f}-{auc_stats['auc_ci_upper']:.2f}])",
)
plt.plot([0, 1], [0, 1], color="black", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - {model_name} ({args.num_shots}-shot) Test Set")
plt.legend(loc="lower right")

# Save the plot
plt.savefig(plot_output_path)
plt.close()
print(f"TabPFN ROC curve plot saved to {plot_output_path}")
