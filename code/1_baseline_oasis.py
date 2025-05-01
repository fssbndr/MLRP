import polars as pl
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# parse command line arguments
parser = argparse.ArgumentParser(description="Run baseline OASIS ROC analysis.")
parser.add_argument("--input", required=True, help="Path to the input parquet data file.")
parser.add_argument("--plot", required=True, help="Path for the output ROC curve plot.")
args = parser.parse_args()

# load the data using the provided path
data = pl.read_parquet(args.input)
data = data.select(
    "Global ICU Stay ID",
    "Mortality in ICU",
    "OASIS ICU Mortality Rate",
)

### SAVE ROC CURVE ###
# Filter out rows where the outcome variable or prediction is null
data = data.filter(pl.col("Mortality in ICU").is_not_null())
data = data.filter(pl.col("OASIS ICU Mortality Rate").is_not_null())

# Define y (true outcome) and y_pred_prob (OASIS score)
y = data.select("Mortality in ICU").to_numpy().flatten()
y_pred_prob = data.select("OASIS ICU Mortality Rate").to_numpy().flatten()

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - OASIS Baseline')
plt.legend(loc="lower right")

# Ensure plot output directory exists
output_dir_plot = os.path.dirname(args.plot)
if output_dir_plot:
    os.makedirs(output_dir_plot, exist_ok=True)

# Save the plot
plt.savefig(args.plot)
plt.close()