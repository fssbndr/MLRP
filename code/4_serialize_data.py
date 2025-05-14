import argparse
import os
import re

import pandas as pd


def load_and_prepare_data(processed_data_path):
    """Loads data from a parquet file and standardizes column names."""
    data_df = pd.read_parquet(processed_data_path)
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


def generate_icu_stay_summary(data_df, global_icu_stay_id_value):
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
        vitals.append(f"GCS: {row['gcs']}")
    if pd.notna(row.get("hr")):
        vitals.append(f"HR: {row['hr']}")
    if pd.notna(row.get("map")):
        vitals.append(f"MAP: {row['map']}")
    if pd.notna(row.get("rr")):
        vitals.append(f"RR: {row['rr']}")
    if pd.notna(row.get("temp_c")):
        vitals.append(f"Temp: {row['temp_c']} C")
    if vitals:
        summary_parts.append(
            f"Worst Vitals in the first 24 hours: {', '.join(vitals)}"
        )

    if pd.notna(row.get("urine_output_ml")):
        summary_parts.append(f"Urine Output: {row['urine_output_ml']} ml")

    if pd.notna(row.get("mechvent")):
        summary_parts.append(f"Mechanical Ventilation: {row['mechvent']}")

    if pd.notna(row.get("mortality_in_icu")):
        summary_parts.append(f"Mortality in ICU: {row['mortality_in_icu']}")

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

    # Define output file paths using os.path.join as per active file context
    txt_output_path = os.path.join(args.output_dir, "serialized_data.txt")

    processed_data_df = load_and_prepare_data(args.input)

    if processed_data_df is not None:
        all_ids = processed_data_df["global_icu_stay_id"].unique()
        with open(txt_output_path, "w") as f:
            for icu_id in all_ids:
                summary = generate_icu_stay_summary(processed_data_df, icu_id)
                f.write(summary + "\n")
        print(f"Successfully generated summaries to {txt_output_path}")
    else:
        print(
            f"Could not generate summaries due to data loading issues from {args.input}"
        )
