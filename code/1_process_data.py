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
parser.add_argument(
    "--hourly",
    action="store_true",
    help="Include hourly aggregations of vital signs (requires at least 8 hours ICU stay).",
)
parser.add_argument(
    "--stats",
    action="store_true",
    help="Include mean, stdev, min, max aggregations of vital signs instead of just max.",
)
args = parser.parse_args()

# Ensure output directory exists
os.makedirs(os.path.dirname(args.output), exist_ok=True)

# load the data using the provided path
source_data = pl.read_parquet(args.input)

# Apply ICU length of stay filter if hourly processing is requested
if args.hourly:
    source_data = source_data.filter(
        pl.col("ICU Length of Stay (days)") > (1 / 3)  # at least 8 hours
    )

data = source_data.select(
    "Global ICU Stay ID",
    *(["Time Relative to Admission (seconds)"] if args.hourly else []),
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

# Handle hourly aggregations if requested
if args.hourly:
    SECONDS_IN_HOUR = 3600
    hourly_vitals = (
        data.with_columns(
            pl.col("Time Relative to Admission (seconds)")
            .floordiv(SECONDS_IN_HOUR)
            .alias("Hour Relative to Admission")
        )
        .group_by("Global ICU Stay ID", "Hour Relative to Admission")
        .agg(
            pl.col("Glasgow coma score total").max().alias("GCS"),
            pl.col("Heart rate").mean().alias("HR"),
            pl.col("Mean arterial pressure").mean().alias("MAP"),
            pl.col("Temperature").mean().alias("Temp (C)"),
            pl.col("Respiratory rate").mean().alias("RR"),
        )
        .pivot(
            index="Global ICU Stay ID",
            on="Hour Relative to Admission",
            values=["GCS", "HR", "MAP", "Temp (C)", "RR"],
            aggregate_function="mean",
            separator=" Hour ",
        )
    )

# Build aggregation list
agg_list = [
    pl.col("Mortality in ICU").first(),
    pl.col("Pre-ICU Length of Stay (days)").min().alias("Pre-ICU LOS (days)"),
    pl.col("Admission Age (years)").min().alias("Age (years)"),
]

# Add vital signs aggregations only if not using hourly data
if not args.hourly:
    vital_signs = [
        ("Glasgow coma score total", "GCS"),
        ("Heart rate", "HR"),
        ("Mean arterial pressure", "MAP"),
        ("Temperature", "Temp (C)"),
        ("Respiratory rate", "RR"),
    ]

    if args.stats:
        # Add comprehensive statistics for each vital sign
        for col_name, alias in vital_signs:
            agg_list.extend(
                [
                    pl.col(col_name).mean().alias(f"{alias} mean"),
                    pl.col(col_name).std().alias(f"{alias} std"),
                    pl.col(col_name).min().alias(f"{alias} min"),
                    pl.col(col_name).max().alias(f"{alias} max"),
                ]
            )
    else:
        # Add simple max aggregation for vital signs
        agg_list.extend(
            [
                pl.col(col_name).max().alias(alias)
                for col_name, alias in vital_signs
            ]
        )

# Add remaining patient aggregations
agg_list.extend(
    [
        pl.col("Urine output").sum().alias("Urine output (ml)"),
        pl.col("Admission Type").first().cast(str).alias("Admission Type"),
        pl.col("Admission Urgency")
        .first()
        .cast(str)
        .alias("Admission Urgency"),
        pl.col("is mechanically ventilated").max().alias("MechVent"),
    ]
)

# Aggregate data for each patient
data = data.group_by("Global ICU Stay ID").agg(agg_list)

# Join hourly aggregations if requested
if args.hourly:
    data = data.join(hourly_vitals, on="Global ICU Stay ID", how="left")

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
