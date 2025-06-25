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
parser.add_argument(
    "--minmax",
    action="store_true",
    help="Include only min, max aggregations of vital signs instead of just max.",
)
parser.add_argument(
    "--forward-fill",
    action="store_true",
    help="Forward-fill missing values in hourly data (only works with --hourly).",
)
parser.add_argument(
    "--hourly-long",
    action="store_true",
    help="Include hourly aggregations in long format (one row per patient per hour).",
)
parser.add_argument(
    "--survival",
    action="store_true",
    help="Include survival analysis format with time_start and time_stop columns.",
)
args = parser.parse_args()

# Ensure output directory exists
os.makedirs(os.path.dirname(args.output), exist_ok=True)

# load the data using the provided path
source_data = pl.read_parquet(args.input)

# Apply ICU length of stay filter if hourly processing is requested
if args.hourly or args.hourly_long or args.survival:
    source_data = source_data.filter(
        pl.col("ICU Length of Stay (days)") > (1 / 3)  # at least 8 hours
    )

# Validate argument combinations
if args.hourly and args.minmax:
    raise ValueError("Cannot combine --hourly and --minmax options")
if args.stats and args.minmax:
    raise ValueError("Cannot combine --stats and --minmax options")
if args.forward_fill and not args.hourly:
    raise ValueError("--forward-fill can only be used with --hourly option")

data = source_data.select(
    "Global ICU Stay ID",
    *(
        ["Time Relative to Admission (seconds)"]
        if args.hourly or args.hourly_long or args.survival
        else []
    ),
    "Mortality in ICU",
    "Pre-ICU Length of Stay (days)",
    "ICU Length of Stay (days)",
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

# Common constants and aggregation expressions
SECONDS_IN_HOUR = 3600

# Patient characteristics aggregations (constant per patient)
patient_characteristics = [
    pl.col("Mortality in ICU").first(),
    pl.col("Pre-ICU Length of Stay (days)").min().alias("Pre-ICU LOS (days)"),
    pl.col("ICU Length of Stay (days)").max().alias("ICU LOS (days)"),
    pl.col("Admission Age (years)").min().alias("Age (years)"),
    pl.col("Admission Type").first().cast(str).alias("Admission Type"),
    pl.col("Admission Urgency").first().cast(str).alias("Admission Urgency"),
    pl.col("is mechanically ventilated").max().alias("MechVent"),
]

# Vital signs aggregations
vital_signs = [
    pl.col("Glasgow coma score total").max().alias("GCS"),
    pl.col("Heart rate").mean().round(1).alias("HR"),
    pl.col("Mean arterial pressure").mean().round(1).alias("MAP"),
    pl.col("Temperature").mean().round(1).alias("Temp (C)"),
    pl.col("Respiratory rate").mean().round(1).alias("RR"),
]

# Handle survival analysis format if requested
if args.survival:
    agg_list = patient_characteristics.copy()
    agg_list.extend(vital_signs)
    agg_list.append(pl.col("Urine output").sum().alias("Urine output (ml)"))

    # Create survival format data
    data = (
        data.with_columns(
            pl.col("Time Relative to Admission (seconds)")
            .floordiv(SECONDS_IN_HOUR)
            .alias("Hour Relative to Admission"),
        )
        .group_by("Global ICU Stay ID", "Hour Relative to Admission")
        .agg(agg_list)
        .join(
            source_data.group_by("Global ICU Stay ID").agg(
                pl.col("ICU Length of Stay (days)").max().alias("ICU_LOS_days"),
                pl.col("Mortality in ICU").first().alias("Final_Mortality"),
            ),
            on="Global ICU Stay ID",
            how="left",
        )
        .with_columns(
            pl.col("Hour Relative to Admission").alias("time_start"),
            pl.when(pl.col("Hour Relative to Admission") == 23)
            .then((pl.col("ICU_LOS_days") * 24))
            .otherwise(pl.col("Hour Relative to Admission") + 1)
            .alias("time_stop"),
            pl.when(
                pl.col("Hour Relative to Admission") == 23,
            )
            .then(pl.col("Final_Mortality"))
            .otherwise(pl.lit(False))
            .alias("Mortality in ICU"),
        )
        .drop("ICU_LOS_days", "Final_Mortality", "Hour Relative to Admission")
        .sort("Global ICU Stay ID", "time_start")
        .with_columns(
            pl.col(
                "Pre-ICU LOS (days)",
                "Age (years)",
                "Admission Type",
                "Admission Urgency",
                "MechVent",
            )
            .forward_fill()
            .over("Global ICU Stay ID"),
        )
    )

    print(f"Survival format data shape: {data.shape}")

# Handle hourly aggregations in long format if requested
elif args.hourly_long:
    agg_list = patient_characteristics.copy()
    agg_list.extend(vital_signs)
    agg_list.append(pl.col("Urine output").sum().alias("Urine output (ml)"))

    # Create hourly aggregated data in long format
    data = (
        data.with_columns(
            pl.col("Time Relative to Admission (seconds)")
            .floordiv(SECONDS_IN_HOUR)
            .alias("Hour Relative to Admission"),
            # Definitively alive until proven dead
            pl.lit(False).alias("Mortality in ICU"),
        )
        .group_by("Global ICU Stay ID", "Hour Relative to Admission")
        .agg(agg_list)
    )

    # Add discharge row for each patient
    discharge_rows = (
        source_data.group_by("Global ICU Stay ID")
        .agg(
            pl.col("ICU Length of Stay (days)").max().alias("ICU_LOS_days"),
            pl.col("Mortality in ICU").first(),
        )
        .with_columns(
            # Convert ICU stay to hours and round up
            (pl.col("ICU_LOS_days") * 24)
            .ceil()
            .cast(float)
            .alias("Hour Relative to Admission"),
        )
        .drop("ICU_LOS_days")
    )

    # Combine hourly data with discharge rows
    data = (
        pl.concat([data, discharge_rows], how="diagonal")
        .sort("Global ICU Stay ID", "Hour Relative to Admission")
        .with_columns(
            pl.col(
                "Pre-ICU LOS (days)",
                "Age (years)",
                "Admission Type",
                "Admission Urgency",
                "MechVent",
            )
            .forward_fill()
            .over("Global ICU Stay ID"),
        )
    )

    print(f"Long format hourly data shape: {data.shape}")

# Handle hourly aggregations if requested
elif args.hourly:
    hourly_vitals = (
        data.with_columns(
            pl.col("Time Relative to Admission (seconds)")
            .floordiv(SECONDS_IN_HOUR)
            .alias("Hour Relative to Admission")
        )
        .group_by("Global ICU Stay ID", "Hour Relative to Admission")
        .agg(vital_signs)
    )

    # Apply forward-fill if requested
    if args.forward_fill:
        hourly_vitals = hourly_vitals.sort(
            "Global ICU Stay ID", "Hour Relative to Admission"
        )
        vital_cols = ["GCS", "HR", "MAP", "Temp (C)", "RR"]
        hourly_vitals = hourly_vitals.with_columns(
            pl.col(col)
            .forward_fill()
            .over("Global ICU Stay ID", order_by="Hour Relative to Admission")
            for col in vital_cols
        )

    hourly_vitals = hourly_vitals.pivot(
        index="Global ICU Stay ID",
        on="Hour Relative to Admission",
        values=["GCS", "HR", "MAP", "Temp (C)", "RR"],
        aggregate_function="mean",
        separator=" Hour ",
    )

    # Build aggregation list
    agg_list = patient_characteristics.copy()
    agg_list.append(pl.col("Urine output").sum().alias("Urine output (ml)"))

    # Aggregate data for each patient
    data = (
        data.group_by("Global ICU Stay ID")
        .agg(agg_list)
        .join(hourly_vitals, on="Global ICU Stay ID", how="left")
    )

else:
    # Build aggregation list
    agg_list = patient_characteristics.copy()

    # Add vital signs aggregations
    vital_signs_tuples = [
        ("Glasgow coma score total", "GCS"),
        ("Heart rate", "HR"),
        ("Mean arterial pressure", "MAP"),
        ("Temperature", "Temp (C)"),
        ("Respiratory rate", "RR"),
    ]

    for col_name, alias in vital_signs_tuples:
        # Add comprehensive statistics for each vital sign
        if args.stats:
            agg_list.extend(
                [
                    pl.col(col_name).mean().alias(f"{alias} mean"),
                    pl.col(col_name).std().alias(f"{alias} std"),
                    pl.col(col_name).min().alias(f"{alias} min"),
                    pl.col(col_name).max().alias(f"{alias} max"),
                ]
            )

        # Add only min and max aggregations for each vital sign
        elif args.minmax:
            agg_list.extend(
                [
                    pl.col(col_name).min().alias(f"{alias} min"),
                    pl.col(col_name).max().alias(f"{alias} max"),
                ]
            )

        else:
            # Add simple max aggregation for vital signs
            agg_list.extend([pl.col(col_name).max().alias(alias)])

    # Add urine output aggregation
    agg_list.append(pl.col("Urine output").sum().alias("Urine output (ml)"))

    # Aggregate data for each patient
    data = data.group_by("Global ICU Stay ID").agg(agg_list)

print(f"Data shape after processing: {data.shape}")

# Add a column for train/test split (80/20) - common for all processing paths
train_test_split = (
    source_data.group_by("Global ICU Stay ID")
    .agg(pl.col("Source Dataset").first())
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

data = data.join(
    train_test_split, on="Global ICU Stay ID", how="left", coalesce=True
)
print(f"Data shape after adding train/test split: {data.shape}")

# save the processed data to a parquet file
data.write_parquet(args.output)

# Print appropriate completion message
print(f"Processed data saved to {args.output}")
