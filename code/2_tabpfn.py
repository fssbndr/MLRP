import polars as pl
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from tabpfn import TabPFNRegressor

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input", required=True, help="Path to the input parquet data file."
)
parser.add_argument(
    "--plot_dir", required=True, help="Path to the output directory for the ROC curve plot."
)
args = parser.parse_args()

# Define output file paths
plot_output_path = os.path.join(args.plot_dir, "tabpfn_roc_curve.png")

# Ensure output directories exist
os.makedirs(args.plot_dir, exist_ok=True)

# load the data using the provided path
data = pl.read_parquet(args.input)

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

# Convert features to numpy array for TabPFN
X = X.to_numpy()

# Instantiate and fit the TabPFN Regressor
regressor = TabPFNRegressor()
regressor.fit(X, y)

### SAVE ROC CURVE ###
# Predict probabilities
y_pred_prob = regressor.predict(X)

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
