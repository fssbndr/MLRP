import argparse
import os
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm
import ollama

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_serialized_data(file_path):
    """Loads serialized summaries and extracts Global ICU Stay ID."""
    summaries = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = re.search(r"Patient ID:\s*(\S+?)\s*\.", line)
            if match:
                try:
                    icu_id = match.group(1)
                    summaries.append(
                        {"global_icu_stay_id": icu_id, "summary_text": line}
                    )
                except ValueError:
                    print(f"Warning: Could not parse ID from line: {line}")
            else:
                print(f"Warning: Could not find 'Patient ID: ...' pattern in line: {line}") # fmt: skip

    if not summaries:
        # Raise an error if no summaries were found for some reason
        raise ValueError(
            f"No summaries could be loaded from {file_path}. "
            "Check file content, format, and Patient ID pattern."
        )
    return pd.DataFrame(summaries)


def load_processed_data(file_path):
    """Loads processed data, including target variable and train/test split."""
    df = pd.read_parquet(file_path, dtype_backend="pyarrow").rename(
        columns={
            "Global ICU Stay ID": "global_icu_stay_id",
            "Mortality in ICU": "mortality_in_icu",
        }
    )

    required_cols = ["global_icu_stay_id", "mortality_in_icu", "split_80_20"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        # Raise an error if columns are missing for some reason
        raise ValueError(f"Error: Missing required columns in processed data: {missing_cols}") # fmt: skip

    return df[required_cols]


def evaluate_model(
    test_df: pd.DataFrame,
    model_name: str,
    train_df: pd.DataFrame,
    num_shots: int,
):
    """Evaluates a given LLM model (via Ollama), optionally with few-shot examples."""
    print(
        f"Evaluating {model_name} with {num_shots}-shot prompting using Ollama..."
    )

    predictions = []
    actual_labels = []
    icu_ids = []

    for _, row in tqdm(
        test_df.iterrows(),
        total=test_df.shape[0],
        desc=f"Evaluating {model_name}",
    ):
        summary_text = row["summary_text"]
        actual_label = row["mortality_in_icu"]

        prompt_parts = []

        # Explain the task
        prompt_parts.append(
            "You are a useful medical assistant. "
            "Your task is to determine whether a patient will die in the ICU based on their summary.\n"
            "You will be given a summary of the patient's ICU stay"
            ""
            if num_shots == 0
            else " and some additional examples.\n"
        )

        # Few-shot examples
        if num_shots > 0 and not train_df.empty:
            died_df = train_df[train_df["mortality_in_icu"] == 1]
            survived_df = train_df[train_df["mortality_in_icu"] == 0]

            n_sample_died = min(num_shots, len(died_df))
            n_sample_survived = min(num_shots, len(survived_df))

            few_shot_list = [
                died_df.sample(n=n_sample_died, random_state=42),
                survived_df.sample(n=n_sample_survived, random_state=42),
            ]

            # Shuffle the combined examples to mix died/survived cases
            few_shot_examples_to_process = (
                pd.concat(few_shot_list)
                .sample(frac=1, random_state=42)
                .reset_index(drop=True)
            )

            for _, fs_row in few_shot_examples_to_process.iterrows():
                prompt_parts.append(fs_row["summary_text"])

        # Actual query
        query_prompt_part = (
            f"ICU Stay Summary:\n{summary_text}\n\n"
            "Your answer must be a floating point number between 0.0 and 1.0, "
            "representing the probability that the patient will die in the ICU. \n"
            "Based on this summary, what is the probability that the patient will die in the ICU? "
            "Include absolutely no additional text or explanation, answer only the number."
        )
        prompt_parts.append(query_prompt_part)

        prompt = "".join(prompt_parts)

        model_inputs = tokenizer(prompt, return_tensors="pt").to(device)
        response = ollama.generate(
            model=model_name, prompt=prompt, options={"num_predict": 100}
        )

        # Ollama response is a dict, actual text is in response['response']
        answer_part = response["response"].strip().lower()
        response_text = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        answer_part = response_text[len(prompt) :].strip().lower()

        predicted_probability = 0.5  # Default to 0.5 (uncertain)
        try:
            # Attempt to extract a float from the answer_part
            # This regex finds the first floating point number or integer.
            match = re.search(r"[-+]?\d*\.\d+|\d+", answer_part)
            if match:
                extracted_value = float(match.group(0))
                # Clamp the value between 0 and 1
                predicted_probability = max(0.0, min(1.0, extracted_value))
            else:
                # Fallback for "yes" / "no" if no number is found, though less ideal
                if "yes" in answer_part:
                    predicted_probability = 1.0
                elif "no" in answer_part:
                    predicted_probability = 0.0
                else:
                    print(
                        f"Warning: Model {model_name} produced an unclear answer for probability: "
                        f"'{answer_part}' for summary ID {row.get('global_icu_stay_id', 'Unknown')}."
                        " Interpreting as 0.5.\n"
                        f"Actual label: {actual_label}\n"
                        f"Full response: {response['response']}"
                    )
        except ValueError:
            print(
                f"Warning: Model {model_name} produced an answer that could not be parsed as float: "
                f"'{answer_part}' for summary ID {row.get('global_icu_stay_id', 'Unknown')}."
                " Interpreting as 0.5.\n"
                f"Actual label: {actual_label}\n"
                f"Full response: {response['response']}"
            )

        predictions.append(predicted_probability)
        actual_labels.append(int(actual_label))
        icu_ids.append(row["global_icu_stay_id"])

    return icu_ids, actual_labels, predictions


# parse command line arguments
parser = argparse.ArgumentParser(
    description="Evaluate a specific LLM on serialized ICU stay summaries."
)
parser.add_argument(
    "--serialized_data_path",
    type=str,
    required=True,
    help="Path to the serialized_data.txt file (train summaries for few-shot, outcomes included in text).",
)
parser.add_argument(
    "--serialized_data_test_path",
    type=str,
    required=True,
    help="Path to the serialized_data_test.txt file (test summaries, outcomes excluded in text).",
)
parser.add_argument(
    "--processed_data_path",
    type=str,
    required=True,
    help="Path to the processed_data.parquet file (for split and target variable).",
)
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Directory to save the evaluation results CSV file.",
)
parser.add_argument(
    "--plot_dir",
    type=str,
    required=True,
    help="Directory to save the ROC curve plot.",
)
parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="File ID of the model to evaluate (e.g., 'qwen_0_5b').",
)
parser.add_argument(
    "--num_shots",
    type=int,
    default=0,
    help="Number of few-shot examples to use (default: 0 for zero-shot).",
)


args = parser.parse_args()

# Define output file paths
output_csv_path = os.path.join(
    args.output_dir, f"llm_{args.model}_{args.num_shots}-shot_results.csv"
)
plot_output_path = os.path.join(
    args.plot_dir, f"llm_{args.model}_{args.num_shots}-shot_roc_curve.png"
)

# Ensure output directories exist
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.plot_dir, exist_ok=True)

# ------------------------------------------------------------------------------
# Load serialized data
serialized_train_df = load_serialized_data(args.serialized_data_path)
serialized_test_df = load_serialized_data(args.serialized_data_test_path)
processed_df = load_processed_data(args.processed_data_path)

# Prepare test_df for evaluation
merged_test_df = pd.merge(
    processed_df,
    serialized_test_df,
    on="global_icu_stay_id",
    how="inner",
)
test_df = merged_test_df[merged_test_df["split_80_20"] == "test"].copy()

# Prepare train_df for few-shot examples
merged_train_df = pd.merge(
    processed_df,
    serialized_train_df,
    on="global_icu_stay_id",
    how="inner",
)
train_df = merged_train_df[merged_train_df["split_80_20"] == "train"].copy()

# Drop rows with missing target variable or summary text
initial_test_rows = len(test_df)
test_df.dropna(subset=["mortality_in_icu", "summary_text"], inplace=True)
rows_dropped_test = initial_test_rows - len(test_df)
if rows_dropped_test > 0:
    print(
        f"Dropped {rows_dropped_test} rows from test set due to missing target or summary text."
    )

initial_train_rows = len(train_df)
train_df.dropna(subset=["mortality_in_icu", "summary_text"], inplace=True)
rows_dropped_train = initial_train_rows - len(train_df)
if rows_dropped_train > 0:
    print(
        f"Dropped {rows_dropped_train} rows from train set for few-shot due to missing target or summary text."
    )

# Define models to evaluate
model_configs = [
    {"name": "llama3.2:3b-instruct-q8_0", "id": "llama_3.2_3b"},
    {"name": "llama3.1:8b-instruct-q8_0", "id": "llama_3.1_8b"},
    # {"name": "qwen2.5:7b-instruct-q8_0",  "id": "qwen_2.5_7b"},
    # {"name": "gemma3:4b-it-q8_0",         "id": "gemma_3_4b"},
    # {"name": "medgemma3:4b-it-q8_0",      "id": "medgemma_3_4b"}, # https://huggingface.co/unsloth/medgemma-4b-it-GGUF
    # {"name": "tabula:8b-q8_0",            "id": "tabula_8b"},     # https://huggingface.co/tensorblock/tabula-8b-GGUF
] # fmt: skip

# Select the target model
model_config = None
for m_config in model_configs:
    if m_config["id"] == args.model:
        model_config = m_config
        break

model_results = {}

################################################################################
# EVALUATE LLM
print(f"--- Starting evaluation for {model_config['name']} ---")

icu_ids, actual_labels, probability_predictions = evaluate_model(
    test_df=test_df,
    model_name=model_config["name"],
    train_df=train_df,
    num_shots=args.num_shots,
)

# Stop execution if no predictions were made
if not actual_labels or not probability_predictions:
    print(f"Evaluation failed for {model_config['name']}.")
    exit()

print(f"--- Finished evaluation for {model_config['name']} ---")
################################################################################

### SAVE RESULTS ###
# Create a DataFrame with patient IDs, actual labels, and predicted probabilities
predictions_df = pd.DataFrame(
    {
        "model_name": model_config["name"],
        "num_shots": args.num_shots,
        "global_icu_stay_id": icu_ids,
        "actual_label": actual_labels,
        "predicted_probability": probability_predictions,  # Changed column name
    }
)
# Set global_icu_stay_id as the index
predictions_df.set_index("global_icu_stay_id", inplace=True)

predictions_df.to_csv(output_csv_path)
print(f"Per-patient evaluation results for {model_config['name']} saved to {output_csv_path}") # fmt: skip

### CALCULATE METRICS ###
# For metrics requiring binary predictions, apply a 0.5 threshold
binary_predictions = [1 if p >= 0.5 else 0 for p in probability_predictions]

# These metrics are for console output / plot titles, not the primary CSV output anymore.
model_metrics = {
    "accuracy": accuracy_score(actual_labels, binary_predictions),
    "auc": roc_auc_score(
        actual_labels, probability_predictions
    ),  # AUC uses probabilities
    "f1": f1_score(actual_labels, binary_predictions, zero_division=0),
    "confusion_matrix": (
        confusion_matrix(actual_labels, binary_predictions).tolist()
    ),
}
print(
    f"Model: {model_config['name']}, "
    f"Num Shots: {args.num_shots}, "
    f"Accuracy: {model_metrics['accuracy']:.2f}, "
    f"AUC: {model_metrics['auc']:.2f}, "
    f"F1 Score: {model_metrics['f1']:.2f}, "
    f"Confusion Matrix: {model_metrics['confusion_matrix']}"
)

### SAVE ROC CURVE ###
# Calculate ROC curve using probabilities
fpr, tpr, thresholds = roc_curve(actual_labels, probability_predictions)
roc_auc_val = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc_val:.2f})")
plt.plot([0, 1], [0, 1], color="black", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - {model_config['name']} Test Set")
plt.legend(loc="lower right")

# Save the plot
plt.savefig(plot_output_path)
plt.close()
