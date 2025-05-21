import argparse
import os

import polars as pl

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input", required=True, help="Path to the input raw parquet data file."
)
parser.add_argument(
    "--output",
    required=True,
    help="Path to the output processed parquet data file.",
)
args = parser.parse_args()

# Ensure output directory exists
os.makedirs(os.path.dirname(args.output), exist_ok=True)

# load the data using the provided path
source_data = pl.read_parquet(args.input)
data = source_data.select(
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
    pl.col("Mortality in ICU").first(),
    pl.col("Pre-ICU Length of Stay (days)").min().alias("Pre-ICU LOS (days)"),
    pl.col("Admission Age (years)").min().alias("Age (years)"),
    pl.col("Glasgow coma score total").max().alias("GCS"),
    pl.col("Heart rate").max().alias("HR"),
    pl.col("Mean arterial pressure").max().alias("MAP"),
    pl.col("Urine output").sum().alias("Urine output (ml)"),
    pl.col("Respiratory rate").max().alias("RR"),
    pl.col("Temperature").max().alias("Temp (C)"),
    pl.col("Admission Type").first().cast(str).alias("Admission Type"),
    pl.col("Admission Urgency").first().cast(str).alias("Admission Urgency"),
    pl.col("is mechanically ventilated").max().alias("MechVent"),
)

print(f"Data shape after processing: {data.shape}")

# Add a column for train/test split (80/20)
data = (
    data.join(
        source_data.group_by("Global ICU Stay ID").agg(
            pl.col("Source Dataset").first()
        ),
        on="Global ICU Stay ID",
        how="left",
        coalesce=True,
    )
    .with_columns(
        pl.when(
            pl.int_range(0, pl.len()).shuffle(seed=42).over("Source Dataset")
            < (0.8 * pl.len().over("Source Dataset")).round().cast(int)
        )
        .then(pl.lit("train"))
        .otherwise(pl.lit("test"))
        .cast(str)
        .alias("split_80_20"),
    )
    .drop("Source Dataset")
)

print(f"Data shape after adding train/test split: {data.shape}")

# save the processed data to a parquet file
data.write_parquet(args.output)

print(f"Processed data saved to {args.output}")
