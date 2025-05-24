import argparse
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import ollama
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

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ------------------------------------------------------------------------------
# region loading data
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


# ------------------------------------------------------------------------------
# region processing data
def _process_row(
    row_tuple,
    model_name: str,
    few_shot_prompt_text: str,
    num_shots: int,
):
    """Processes a single row for LLM evaluation."""
    _, row = row_tuple  # row_tuple is (index, series) from df.iterrows()
    summary_text = row["summary_text"]
    actual_label = row["mortality_in_icu"]
    icu_id = row["global_icu_stay_id"]

    prompt_parts = []
    prompt_parts.append(
        "You are a useful medical assistant. "
        "Your task is to determine whether a patient will die in the ICU based on their summary.\n"
        "You will be given a summary of the patient's ICU stay"
    )
    if num_shots > 0 and few_shot_prompt_text:
        prompt_parts.append(" and some additional examples.\n")
        prompt_parts.append(few_shot_prompt_text)
    else:
        prompt_parts.append(".\n")  # Ensure correct punctuation if no few-shot

    query_prompt_part = (
        f"ICU Stay Summary:\n{summary_text}\n\n"
        "Your answer must be a floating point number between 0.0 and 1.0, "
        "representing the probability that the patient will die in the ICU. \n"
        "Based on this summary, what is the probability that the patient will die in the ICU? "
        "Include absolutely no additional text or explanation, answer only the number."
    )
    prompt_parts.append(query_prompt_part)
    prompt = "".join(prompt_parts)

    predicted_probability = None  # Default to None (uncertain)
    full_response_text = ""

    try:
        response = ollama.generate(
            model=model_name, prompt=prompt, options={"num_predict": 100}
        )
        answer_part = response["response"].strip().lower()
        full_response_text = response["response"]

        match = re.search(r"[-+]?\d*\.\d+|\d+", answer_part)
        if match:
            extracted_value = float(match.group(0))
            predicted_probability = max(0.0, min(1.0, extracted_value))
        else:
            if "yes" in answer_part:
                predicted_probability = 1.0
            elif "no" in answer_part:
                predicted_probability = 0.0
            else:
                print(
                    f"Warning: Model {model_name} produced an unclear answer for probability: "
                    f"'{answer_part}' for summary ID {icu_id}."
                    " Interpreting as 0.5.\n"
                    f"Actual label: {actual_label}\n"
                    f"Full response: {full_response_text}"
                )
    except ValueError:
        print(
            f"Warning: Model {model_name} produced an answer that could not be parsed as float: "
            f"'{answer_part}' for summary ID {icu_id}."
            " Interpreting as 0.5.\n"
            f"Actual label: {actual_label}\n"
            f"Full response: {full_response_text}"
        )
    except Exception as e:
        print(
            f"Error during Ollama generation for model {model_name}, "
            f"ID {icu_id}: {e}"
        )

    return icu_id, int(actual_label), predicted_probability


def evaluate_model(
    test_df: pd.DataFrame,
    model_name: str,
    train_df: pd.DataFrame,
    num_shots: int,
    max_workers: int = 10,
):
    """Evaluates a given LLM model (via Ollama), optionally with few-shot examples, using parallel processing."""
    print(f"Evaluating {model_name} with {num_shots}-shot prompting...")

    # Precompute few-shot examples prompt part
    few_shot_prompt_text = ""
    if num_shots > 0 and not train_df.empty:
        died_df = train_df[train_df["mortality_in_icu"] == 1]
        survived_df = train_df[train_df["mortality_in_icu"] == 0]

        n_sample_died = min(num_shots, len(died_df))
        n_sample_survived = min(num_shots, len(survived_df))

        if n_sample_died < num_shots:
            print(
                f"Warning: Only {n_sample_died} examples available for 'died' class. "
                f"Using {n_sample_died} instead of {num_shots}."
            )
        if n_sample_survived < num_shots:
            print(
                f"Warning: Only {n_sample_survived} examples available for 'survived' class. "
                f"Using {n_sample_survived} instead of {num_shots}."
            )

        few_shot_list = []
        if n_sample_died > 0:
            few_shot_list.append(
                died_df.sample(n=n_sample_died, random_state=SEED)
            )
        if n_sample_survived > 0:
            few_shot_list.append(
                survived_df.sample(n=n_sample_survived, random_state=SEED)
            )

        few_shot_examples_to_process = (
            pd.concat(few_shot_list)
            .sample(frac=1, random_state=SEED)
            .reset_index(drop=True)
        )
        temp_prompt_parts = []
        for _, fs_row in few_shot_examples_to_process.iterrows():
            temp_prompt_parts.append(fs_row["summary_text"])
        few_shot_prompt_text = "".join(temp_prompt_parts)

    results_list = []
    tasks = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for row_tuple in test_df.iterrows():
            tasks.append(
                executor.submit(
                    _process_row,
                    row_tuple,
                    model_name,
                    few_shot_prompt_text,
                    num_shots,
                )
            )

        for future in tqdm(
            as_completed(tasks),
            total=len(tasks),
            desc=f"Evaluating {model_name}",
        ):
            # result = (icu_id, actual_label, predicted_probability)
            results_list.append(future.result())

    icu_ids = [res[0] for res in results_list]
    actual_labels = [res[1] for res in results_list]
    predictions = [res[2] for res in results_list]

    return icu_ids, actual_labels, predictions


# ------------------------------------------------------------------------------
# region main
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

# Filter out None predictions and prepare data for metrics and output
total_predictions = len(probability_predictions)
final_icu_ids = []
final_actual_labels = []
final_probability_predictions = []

if total_predictions > 0:  # Ensure lists are not empty before iterating
    for i in range(total_predictions):
        if probability_predictions[i] is not None:
            final_icu_ids.append(icu_ids[i])
            final_actual_labels.append(actual_labels[i])
            final_probability_predictions.append(probability_predictions[i])

none_predictions_count = total_predictions - len(final_probability_predictions)

if none_predictions_count > 0:
    print(
        f"Warning: Dropped {none_predictions_count} predictions due to None probability."
    )


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
        "predicted_probability": probability_predictions,
    }
)
# Set global_icu_stay_id as the index
predictions_df.set_index("global_icu_stay_id", inplace=True)

predictions_df.to_csv(output_csv_path)
print(f"Per-patient evaluation results for {model_config['name']} saved to {output_csv_path}") # fmt: skip

### CALCULATE METRICS ###
# For metrics requiring binary predictions, apply a 0.5 threshold
# final_probability_predictions is guaranteed not to have None here.
binary_predictions = [
    1 if p >= 0.5 else 0 for p in final_probability_predictions
]

# Calculate metrics only if there's valid data after filtering
model_metrics = {
    "accuracy": accuracy_score(final_actual_labels, binary_predictions),
    "auc": roc_auc_score(final_actual_labels, final_probability_predictions),
    "f1": f1_score(final_actual_labels, binary_predictions, zero_division=0),
    "cm": confusion_matrix(final_actual_labels, binary_predictions).tolist(),
    "missingness": none_predictions_count / total_predictions,
}
print(
    f"Model: {model_config['name']}, "
    f"Num Shots: {args.num_shots}, "
    f"missing Predictions: {model_metrics['missingness']:.1%}, "
    f"Accuracy: {model_metrics['accuracy']:.2f}, "
    f"AUC: {model_metrics['auc']:.2f}, "
    f"F1 Score: {model_metrics['f1']:.2f}, "
    f"Confusion Matrix: {model_metrics['cm']}"
)

### SAVE ROC CURVE ###
# Calculate ROC curve using filtered probabilities
fpr, tpr, thresholds = roc_curve(
    final_actual_labels, final_probability_predictions
)
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
