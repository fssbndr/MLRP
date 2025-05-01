import polars as pl
import statsmodels.api as sm
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input", required=True, help="Path to the input parquet data file."
)
parser.add_argument(
    "--output_dir", required=True, help="Path to the output directory for summary and processed data."
)
parser.add_argument(
    "--plot_dir", required=True, help="Path to the output directory for the ROC curve plot."
)
args = parser.parse_args()

# Define output file paths
summary_output_path = os.path.join(args.output_dir, "baseline_logistic_summary.txt")
parquet_output_path = os.path.join(args.output_dir, "baseline_logistic_processed.parquet")
plot_output_path = os.path.join(args.plot_dir, "baseline_logistic_roc_curve.png")

# Ensure output directories exist
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.plot_dir, exist_ok=True)

# load the data using the provided path
data = pl.read_parquet(args.input)
data = data.select(
    "Global ICU Stay ID",
    "Mortality in ICU",
    "Pre-ICU Length of Stay (days)",
    "Admission Age (years)",
    "Glasgow coma score total",
    "Heart rate",
    pl.coalesce(
        pl.col("Invasive mean arterial pressure"),
        pl.col("Non-invasive mean arterial pressure"),
        (
            2 * pl.col("Invasive systolic arterial pressure")
            + pl.col("Invasive diastolic arterial pressure")
        )
        / 3,
        (
            2 * pl.col("Non-invasive systolic arterial pressure")
            + pl.col("Non-invasive diastolic arterial pressure")
        )
        / 3,
    ).alias("Mean arterial pressure"),
    pl.sum_horizontal(
        "Fluid output urine in and out urethral catheter",
        "Fluid output urine nephrostomy",
        "Urine output",
    ).alias("Urine output"),
    "Respiratory rate",
    "Temperature",
    "Admission Type",
    "Admission Urgency",
    "is mechanically ventilated",
)

# aggregate the mean / max / min values for each patient
data = data.group_by("Global ICU Stay ID").agg(
    pl.col("Mortality in ICU").first(),  # Keep mortality for outcome
    pl.col("Pre-ICU Length of Stay (days)").min().alias("Pre-ICU LOS (days)"),
    pl.col("Admission Age (years)").min().alias("Age (years)"),
    pl.col("Glasgow coma score total").max().alias("GCS"),
    pl.col("Heart rate").max().alias("HR"),
    pl.col("Mean arterial pressure").max().alias("MAP"),
    pl.col("Urine output").sum().alias("Urine output (ml)"),
    pl.col("Respiratory rate").max().alias("RR"),
    pl.col("Temperature").max().alias("Temp (C)"),
    pl.col("Admission Type").first().alias("Admission Type"),
    pl.col("Admission Urgency").first().alias("Admission Urgency"),
    pl.col("is mechanically ventilated").max().alias("MechVent"),
)

# save the processed data to a parquet file
data.write_parquet(parquet_output_path)

# filter out rows where the outcome variable is null
data = data.filter(pl.col("Mortality in ICU").is_not_null())

# fit the logistic regression model with ICU mortality as the outcome
X = data.select(
    "Pre-ICU LOS (days)",
    "Age (years)",
    "GCS",
    "HR",
    "MAP",
    "Urine output (ml)",
    "RR",
    "Temp (C)",
    "Admission Type",  # Categorical
    "Admission Urgency",  # Categorical
    "MechVent",
)
y = data.select("Mortality in ICU").to_numpy().flatten()

# Fill null values in X before converting to numpy or creating dummies
X = X.fill_null(0)

# Convert categorical columns to dummy variables
categorical_cols = ["Admission Type", "Admission Urgency"]
X = X.to_dummies(columns=categorical_cols, drop_first=True)

# Now X should be all numeric
X = sm.add_constant(X.to_numpy())
model = sm.Logit(y, X)
result = model.fit(disp=0)

### SAVE SUMMARY ###
# Write summary to file using the constructed path
with open(summary_output_path, "w") as f:
    f.write(str(result.summary()))

### SAVE ROC CURVE ###
# Predict probabilities
y_pred_prob = result.predict(X)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="black", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")

# Save the plot
plt.savefig(plot_output_path)
plt.close()
