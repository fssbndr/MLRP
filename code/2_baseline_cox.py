import argparse
import os

import matplotlib.pyplot as plt
import polars as pl
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from lifelines import CoxPHFitter
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
summary_output_path = os.path.join(args.output_dir, f"baseline_cox{suffix}_summary.txt")
plot_output_path = os.path.join(args.plot_dir, f"baseline_cox{suffix}_roc_curve.png")
csv_output_path = os.path.join(args.output_dir, f"baseline_cox{suffix}_results.csv")

# Check for ICU duration column - try different possible names
duration_col = None
possible_duration_cols = [
    "ICU Stay Duration (days)",
    "ICU LOS (days)",
    "LOS ICU (days)",
]
for col in possible_duration_cols:
    if col in data.columns:
        duration_col = col
        break

print(f"Using duration column: {duration_col}")

# Get train/test masks from the 'split_80_20' column
train_mask = (data["split_80_20"] == "train").to_numpy()
test_mask = (data["split_80_20"] == "test").to_numpy()

# Split data
train_data = data.filter(pl.Series(train_mask))
test_data = data.filter(pl.Series(test_mask))

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

print(f"Using {len(feature_columns)} features for Cox PH model{suffix}")

# Prepare survival data for training
PREDICTION_HORIZON_DAYS = 3
train_durations = train_data[duration_col].to_numpy()
train_events = train_data["Mortality in ICU"].to_numpy().astype(bool)

# Remove any rows with null durations
valid_train_mask = ~np.isnan(train_durations)
train_durations = train_durations[valid_train_mask]
train_events = train_events[valid_train_mask]

print(
    f"Training data: {len(train_durations)} samples, {train_events.sum()} events"
)

# Prepare feature matrix for Cox model
train_features_df = train_data.select(
    [col for col in feature_columns if col in data.columns]
)
train_features_df = train_features_df.filter(pl.Series(valid_train_mask))

# Fill null values and handle categorical variables
categorical_cols = ["Admission Type", "Admission Urgency", "MechVent"]
train_features_df = train_features_df.with_columns(
    [
        pl.when(pl.col(col).is_null())
        .then(pl.lit("Unknown"))
        .otherwise(pl.col(col))
        .cast(str)
        .alias(col)
        for col in categorical_cols
        if col in train_features_df.columns
    ]
)
train_features_df = train_features_df.fill_null(0)

# Convert to pandas for lifelines
train_features_pd = train_features_df.to_pandas()

# Create dummy variables for categorical columns using pandas
for col in categorical_cols:
    if col in train_features_pd.columns:
        dummies = pd.get_dummies(
            train_features_pd[col], prefix=col, drop_first=True
        )
        train_features_pd = train_features_pd.drop(columns=[col])
        train_features_pd = pd.concat([train_features_pd, dummies], axis=1)

# Add duration and event columns
train_features_pd["duration_censored"] = np.minimum(
    train_durations, PREDICTION_HORIZON_DAYS
)
train_features_pd["event_censored"] = train_events & (
    train_durations <= PREDICTION_HORIZON_DAYS
)

print(
    f"Censored training data: {train_features_pd['event_censored'].sum()} events within {PREDICTION_HORIZON_DAYS} days"
)

################################################################################
# COX PROPORTIONAL HAZARDS MODEL
# Fit Cox PH model on training data
cph = CoxPHFitter(penalizer=0.1)
cph.fit(
    train_features_pd,
    duration_col="duration_censored",
    event_col="event_censored",
)

print(f"Fitted Cox PH model. Concordance index: {cph.concordance_index_:.3f}")

# Prepare test data
test_durations = test_data[duration_col].to_numpy()
test_events = test_data["Mortality in ICU"].to_numpy()
test_icu_ids = test_data["Global ICU Stay ID"].to_numpy()

# Remove any rows with null durations from test set
valid_test_mask = ~np.isnan(test_durations)
test_durations = test_durations[valid_test_mask]
test_events = test_events[valid_test_mask]
test_icu_ids = test_icu_ids[valid_test_mask]

print(
    f"Test data: {len(test_durations)} samples, {test_events.sum()} total events"
)

# Prepare test features
test_features_df = test_data.select(
    [col for col in feature_columns if col in data.columns]
)
test_features_df = test_features_df.filter(pl.Series(valid_test_mask))

# Apply same preprocessing as training data
test_features_df = test_features_df.with_columns(
    [
        pl.when(pl.col(col).is_null())
        .then(pl.lit("Unknown"))
        .otherwise(pl.col(col))
        .cast(str)
        .alias(col)
        for col in categorical_cols
        if col in test_features_df.columns
    ]
)
test_features_df = test_features_df.fill_null(0)

# Convert to pandas and create dummy variables
test_features_pd = test_features_df.to_pandas()

# Ensure same dummy variables as training
train_feature_columns = [
    col
    for col in train_features_pd.columns
    if col not in ["duration_censored", "event_censored"]
]

for col in categorical_cols:
    if col in test_features_pd.columns:
        test_dummies = pd.get_dummies(
            test_features_pd[col], prefix=col, drop_first=True
        )
        test_features_pd = test_features_pd.drop(columns=[col])
        test_features_pd = pd.concat([test_features_pd, test_dummies], axis=1)

# Ensure test features match training features exactly
for col in train_feature_columns:
    if col not in test_features_pd.columns:
        test_features_pd[col] = 0

# Keep only the columns that were in training and in the same order
test_features_processed = test_features_pd[train_feature_columns].fillna(0)

# Predict survival probabilities at prediction horizon
survival_funcs = cph.predict_survival_function(test_features_processed)

mortality_probs = []
for i in range(len(test_features_processed)):
    patient_survival = survival_funcs.iloc[:, i]

    # Find survival probability at prediction horizon
    if PREDICTION_HORIZON_DAYS in patient_survival.index:
        survival_prob = patient_survival.loc[PREDICTION_HORIZON_DAYS]
    else:
        # Use closest time point
        times = patient_survival.index
        valid_times = times[times <= PREDICTION_HORIZON_DAYS]
        if len(valid_times) > 0:
            closest_time = valid_times[-1]
            survival_prob = patient_survival.loc[closest_time]
        else:
            survival_prob = 1.0  # No events before horizon

    mortality_prob = 1.0 - survival_prob
    mortality_probs.append(max(0.0, min(1.0, mortality_prob)))

# Prepare actual labels for test set
y_test_np = []
y_pred_prob_test = []

for i, (duration, event, mort_prob) in enumerate(
    zip(test_durations, test_events, mortality_probs)
):
    # Actual label: True if patient died within prediction horizon days
    if event and duration <= PREDICTION_HORIZON_DAYS:
        actual_mortality = 1
    else:
        actual_mortality = 0
    y_test_np.append(actual_mortality)
    y_pred_prob_test.append(mort_prob)

y_test_np = np.array(y_test_np)
y_pred_prob_test = np.array(y_pred_prob_test)

print(
    f"Test set {PREDICTION_HORIZON_DAYS}-day mortality events: {y_test_np.sum()} out of {len(y_test_np)}"
)
print(
    f"Test set {PREDICTION_HORIZON_DAYS}-day mortality rate: {y_test_np.mean():.3f}"
)
print(f"Mean predicted mortality probability: {y_pred_prob_test.mean():.3f}")

################################################################################

### SAVE SUMMARY ###
summary_stats = {
    "n_train": len(train_features_pd),
    "n_test": len(test_durations),
    "n_events_train": int(train_features_pd["event_censored"].sum()),
    "n_events_test": int(y_test_np.sum()),
    "concordance_index": (
        cph.concordance_index_ if hasattr(cph, "concordance_index_") else 0.5
    ),
    "prediction_horizon_days": PREDICTION_HORIZON_DAYS,
}

with open(summary_output_path, "w") as f:
    f.write("Cox Proportional Hazards Survival Analysis Summary\n")
    f.write("=" * 55 + "\n\n")
    f.write(f"Training samples: {summary_stats['n_train']}\n")
    f.write(f"Test samples: {summary_stats['n_test']}\n")
    f.write(
        f"Training events (≤{PREDICTION_HORIZON_DAYS} days): {summary_stats['n_events_train']}\n"
    )
    f.write(
        f"Test events (≤{PREDICTION_HORIZON_DAYS} days): {summary_stats['n_events_test']}\n"
    )
    f.write(f"Concordance index: {summary_stats['concordance_index']:.3f}\n")
    f.write(f"Prediction horizon: {PREDICTION_HORIZON_DAYS} days\n\n")

    # Add model summary
    f.write("Cox PH Model Summary:\n")
    f.write("-" * 30 + "\n")
    f.write(str(cph.summary))

### CALCULATE METRICS ###
# Calculate AUC with bootstrapping
auc_stats = bootstrap_auc(
    y_test_np, y_pred_prob_test, n_bootstrap=100, random_state=42
)

# Calculate ROC curve using test data
fpr, tpr, thresholds = roc_curve(y_test_np, y_pred_prob_test)
roc_auc = auc(fpr, tpr)

### SAVE RESULTS TO CSV ###
predictions_df = pl.DataFrame(
    {
        "model_name": "lifelines.CoxPHFitter",
        "model_args": f"baseline_cox{suffix}",
        "num_shots": 0,  # Baseline models don't use shots
        "auc_ci_lower": auc_stats["auc_ci_lower"],
        "auc_ci_upper": auc_stats["auc_ci_upper"],
        "global_icu_stay_id": test_icu_ids,
        "actual_label": y_test_np,
        "predicted_probability": y_pred_prob_test,
    }
)

predictions_df.write_csv(csv_output_path)
print(f"Cox PH{suffix} results saved to {csv_output_path}")

### SAVE ROC CURVE ###
plt.figure(figsize=(10, 8))

# Plot ROC curve
plt.subplot(2, 1, 1)
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
plt.title(
    f"ROC Curve - Cox PH {PREDICTION_HORIZON_DAYS}-day Mortality Prediction"
)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

# Plot hazard ratios
plt.subplot(2, 1, 2)
hazard_ratios = np.exp(cph.params_)
significant_features = cph.summary[cph.summary["p"] < 0.05].index

if len(significant_features) > 0:
    sig_hazard_ratios = hazard_ratios[significant_features]
    plt.barh(range(len(sig_hazard_ratios)), sig_hazard_ratios.values)
    plt.yticks(
        range(len(sig_hazard_ratios)),
        [f.replace("_", " ") for f in sig_hazard_ratios.index],
    )
    plt.axvline(
        x=1, color="red", linestyle="--", alpha=0.7, label="HR = 1 (no effect)"
    )
    plt.xlabel("Hazard Ratio")
    plt.title("Significant Hazard Ratios (p < 0.05)")
    plt.legend()
else:
    plt.text(
        0.5,
        0.5,
        "No significant features (p < 0.05)",
        ha="center",
        va="center",
        transform=plt.gca().transAxes,
    )
    plt.title("Hazard Ratios")

plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(plot_output_path, dpi=300, bbox_inches="tight")
plt.close()

print(
    f"Cox PH{suffix} analysis completed. AUC: {auc_stats['auc']:.3f} [{auc_stats['auc_ci_lower']:.3f}-{auc_stats['auc_ci_upper']:.3f}]"
)
print(f"Concordance Index: {cph.concordance_index_:.3f}")
print(f"Plot saved to {plot_output_path}")
