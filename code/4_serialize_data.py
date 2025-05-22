import argparse
import os
import re

import pandas as pd


def load_and_prepare_data(processed_data_path):
    """Loads data from a parquet file and standardizes column names."""
    data_df = pd.read_parquet(processed_data_path, dtype_backend="pyarrow")
    data_df.columns = [
        col.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
        .replace(".", "")
        for col in data_df.columns
    ]
    return data_df


def clean_note(note_text):
    """Cleans the generated note text."""
    if not isinstance(note_text, str):
        note_text = str(note_text)

    # Consolidate multiple spaces/tabs to a single space
    note = re.sub(r"[ \t]+", " ", note_text)
    # Clean up spaces around newlines
    note = re.sub(r"\s*\n\s*", "\n", note)
    # Limit consecutive newlines to a maximum of two (one blank line)
    note = re.sub(r"\n{3,}", "\n\n", note)
    # Remove leading/trailing whitespace from the whole string
    note = note.strip()

    # Remove spaces before common punctuation marks
    note = re.sub(r"\s+\.", ".", note)
    note = re.sub(r"\s+,", ",", note)
    note = re.sub(r"\s+:", ":", note)
    note = re.sub(r"\s+;", ";", note)

    # Replace multiple periods (e.g., "end...") with a single period
    note = re.sub(r"\.{2,}", ".", note)
    return note


def generate_icu_stay_summary(
    data_df, global_icu_stay_id_value, include_outcome: bool
):
    """Generates a textual summary for a given ICU stay ID."""
    patient_data = data_df[
        data_df["global_icu_stay_id"] == global_icu_stay_id_value
    ]
    if patient_data.empty:
        return f"No data found for ICU Stay ID: {global_icu_stay_id_value}"

    row = patient_data.iloc[0]

    summary_parts = []
    summary_parts.append(f"Patient ID: {global_icu_stay_id_value}")

    if pd.notna(row["age_years"]):
        summary_parts.append(f"Age: {row['age_years']} years")

    admission_parts = []
    if pd.notna(row.get("admission_type")):
        admission_parts.append(str(row["admission_type"]))
    if pd.notna(row.get("admission_urgency")):
        admission_parts.append(f"({row['admission_urgency']})")
    if admission_parts:
        summary_parts.append(
            f"Admission Type (Urgency): {' '.join(admission_parts)}"
        )

    if pd.notna(row.get("pre_icu_los_days")):
        try:
            los_days = float(row["pre_icu_los_days"])
            summary_parts.append(f"Pre-ICU Length of Stay: {los_days:.2f} days")
        except (ValueError, TypeError):
            summary_parts.append(
                f"Pre-ICU Length of Stay: {row['pre_icu_los_days']}"
            )

    vitals = []
    if pd.notna(row.get("gcs")):
        vitals.append(f"Glasgow coma score: {row['gcs']:.0f}")
    if pd.notna(row.get("hr")):
        vitals.append(f"Heart rate: {row['hr']:.0f}")
    if pd.notna(row.get("map")):
        vitals.append(f"Mean arterial pressure: {row['map']:.0f}")
    if pd.notna(row.get("rr")):
        vitals.append(f"Respiratory rate: {row['rr']:.0f}")
    if pd.notna(row.get("temp_c")):
        vitals.append(f"Temperature: {row['temp_c']:.1f} C")
    if vitals:
        summary_parts.append(
            f"Worst Vitals in the first 24 hours: {', '.join(vitals)}"
        )

    if pd.notna(row.get("urine_output_ml")):
        summary_parts.append(f"Urine Output: {row['urine_output_ml']:.0f} ml")

    if pd.notna(row.get("mechvent")):
        summary_parts.append(
            f"Mechanical Ventilation: {'Yes' if row['mechvent'] else 'No'}"
        )

    if include_outcome:
        if pd.notna(row.get("mortality_in_icu")):
            summary_parts.append(
                f"Mortality in ICU: {'Yes' if row['mortality_in_icu'] else 'No'}"
            )

    note = ". ".join(filter(None, summary_parts))
    if note and note != "No summary data available.":
        note += "."
    elif not note:
        note = "No summary data available."

    return clean_note(note)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate serialized ICU stay summaries."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the processed data file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save the serialized summaries TXT file.",
    )
    args = parser.parse_args()

    # Define output file paths
    txt_output_path_train = os.path.join(args.output_dir, "serialized_data.txt")
    txt_output_path_test = os.path.join(args.output_dir, "serialized_data_test.txt") # fmt: skip

    os.makedirs(args.output_dir, exist_ok=True)

    source_df = load_and_prepare_data(args.input)
    train_source_df = source_df[source_df["split_80_20"] == "train"].copy()
    test_source_df = source_df[source_df["split_80_20"] == "test"].copy()

    # Generate serialized_data.txt (training set, with outcome)
    train_ids = train_source_df["global_icu_stay_id"].unique()
    with open(txt_output_path_train, "w") as f:
        for icu_id in train_ids:
            summary = generate_icu_stay_summary(
                train_source_df, icu_id, include_outcome=True
            )
            f.write(summary + "\n")
    print(
        "Successfully generated training summaries (with outcome) "
        f"to {txt_output_path_train}"
    )

    # Generate serialized_data_test.txt (test set, without outcome)
    test_ids = test_source_df["global_icu_stay_id"].unique()
    with open(txt_output_path_test, "w") as f:
        for icu_id in test_ids:
            summary = generate_icu_stay_summary(
                test_source_df, icu_id, include_outcome=False
            )
            f.write(summary + "\n")
    print(
        "Successfully generated test summaries (without outcome) "
        f"to {txt_output_path_test}"
    )
