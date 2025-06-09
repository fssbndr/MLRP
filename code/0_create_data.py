import argparse
import os

import polars as pl
import yaml

SECONDS_IN_A_DAY = 86400

STAY_ID = "Global ICU Stay ID"
TIME_COL = "Time Relative to Admission (seconds)"

parser = argparse.ArgumentParser(description="Create the combined dataset.")
parser.add_argument(
    "--config", required=True, help="Path to the config YAML file."
)
parser.add_argument(
    "--output", required=True, help="Path for the output parquet file."
)
args = parser.parse_args()

# load the config file using the provided path
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

# load the data using paths from config
reprodicu_path = config["reprodicu_demo"]
info = pl.scan_parquet(reprodicu_path + "patient_information.parquet")
vitals = pl.scan_parquet(reprodicu_path + "timeseries_vitals.parquet")
labs = pl.scan_parquet(reprodicu_path + "timeseries_labs.parquet")
resp = pl.scan_parquet(reprodicu_path + "timeseries_respiratory.parquet")
inout = pl.scan_parquet(reprodicu_path + "timeseries_intakeoutput.parquet")

OASIS = pl.scan_parquet(
    config["reprodicu_path"]
    + "PRECALCULATED_CONCEPTS/"
    + "SCORES/"
    + "OASIS_worst_in_first_24h.parquet"
).select(
    STAY_ID,
    "OASIS Score",
    pl.col("oasis_mortality_rate_icu").alias("OASIS ICU Mortality Rate"),
    pl.col("oasis_mortality_rate").alias("OASIS Hospital Mortality Rate"),
)

VENTILATION = pl.scan_parquet(
    config["reprodicu_path"]
    + "MAGIC_CONCEPTS/"
    + "VENTILATION_DURATION.parquet"
)

# combine the reprodICU versions demo data of MIMIC-III and MIMIC-IV into one
# singular dataframe
# ID = info.filter(
#     pl.col("Source Dataset").is_in(["MIMIC-III", "MIMIC-IV", "eICU-CRD"])
# ).select("Global ICU Stay ID")
ID = info.select("Global ICU Stay ID")

data = (
    info.join(vitals, on=STAY_ID, how="left", coalesce=True)
    .join(labs, on=[STAY_ID, TIME_COL], how="full", coalesce=True)
    .join(resp, on=[STAY_ID, TIME_COL], how="full", coalesce=True)
    .join(inout, on=[STAY_ID, TIME_COL], how="full", coalesce=True)
    .join(OASIS, on=STAY_ID, how="left", coalesce=True)
    .join(ID, on=STAY_ID, how="right")
)

# add a binary variable for whether the patient was ventilated
vent = data.join(
    VENTILATION.filter(
        pl.col("Ventilation Type").is_in(
            ["invasive ventilation", "non-invasive ventilation", "tracheostomy"]
        )
        | pl.col("Ventilation Type").is_null()
    ).select(
        STAY_ID,
        "Ventilation Start Relative to Admission (seconds)",
        "Ventilation End Relative to Admission (seconds)",
    ),
    on=STAY_ID,
    how="left",
    coalesce=True,
).select(
    STAY_ID,
    TIME_COL,
    pl.when(
        pl.col(TIME_COL).is_between(
            pl.col("Ventilation Start Relative to Admission (seconds)"),
            pl.col("Ventilation End Relative to Admission (seconds)"),
            closed="both",
        )
    )
    .then(True)
    .otherwise(False)
    .alias("is mechanically ventilated"),
)

data = data.join(vent, on=[STAY_ID, TIME_COL], how="left", coalesce=True)

# filter the data to only include the first 24 hours of each ICU stay
data = data.filter(
    pl.col(TIME_COL).is_between(0, SECONDS_IN_A_DAY, closed="both")
).with_columns(pl.col(TIME_COL).clip(0, SECONDS_IN_A_DAY - 1))

# Ensure output directory exists
output_dir = os.path.dirname(args.output)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

# write the data to the specified output parquet file
data.collect().write_parquet(args.output, use_pyarrow=True)

print(f"Data saved to {args.output}")
