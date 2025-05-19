import argparse
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    """Evaluates a given LLM model, optionally with few-shot examples."""
    print(f"Evaluating {model_name} with {num_shots}-shot prompting...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad_token_id if not set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    predictions = []
    actual_labels = []
    icu_ids = []
    device = model.device

    for _, row in test_df.iterrows():
        summary_text = row["summary_text"]
        actual_label = row["mortality_in_icu"]

        prompt_parts = []

        # Explain the task
        prompt_parts.append(
            "You are a medical assistant. "
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
            n_sample_died = min(num_shots, len(survived_df))
            
            few_shot_list = [
                died_df.sample(n=n_sample_died, random_state=42),
                survived_df.sample(n=n_sample_died, random_state=42),
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
            "Your answer must be exactly 'Yes' or 'No', no numbers. "
            "Include no additional text or explanation.\n"
            "Based on this summary, will the patient die in the ICU? "
        )
        prompt_parts.append(query_prompt_part)

        prompt = "".join(prompt_parts)

        model_inputs = tokenizer(prompt, return_tensors="pt").to(device)
        generated_ids = model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,  # Pass attention_mask
            max_new_tokens=15,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Answer comes after the full prompt (i.e. examples and actual query).
        # -> slice to get only the generated answer part
        response_text = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        answer_part = response_text[len(prompt) :].strip().lower()

        predicted_label = 0  # Default to No
        if "yes" in answer_part:
            predicted_label = 1
        elif "no" in answer_part:
            predicted_label = 0
        else:
            print(
                f"Warning: Model {model_name} produced an unclear answer: "
                f"'{answer_part}' for summary ID {row.get('global_icu_stay_id', 'Unknown')}."
                " Interpreting as 'No'.\n"
                f"Actual label: {actual_label}\n"
                f"Full response: {response_text}"
            )

        predictions.append(predicted_label)
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
    {"name": "Qwen/Qwen2.5-0.5B-Instruct",       "id": "qwen_0_5b"},
    {"name": "meta-llama/Llama-3.2-1B-Instruct", "id": "llama_3_2_1b"},
    {"name": "Qwen/Qwen2.5-1.5B-Instruct",       "id": "qwen_1_5b"},
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

icu_ids, actual_labels, predictions = evaluate_model(
    test_df=test_df,
    model_name=model_config["name"],
    train_df=train_df,
    num_shots=args.num_shots,
)

# Stop execution if no predictions were made
if not actual_labels or not predictions:
    print(f"Evaluation failed for {model_config['name']}.")
    exit()

print(f"--- Finished evaluation for {model_config['name']} ---")
################################################################################

### SAVE RESULTS ###
# Create a DataFrame with patient IDs, actual labels, and predictions
predictions_df = pd.DataFrame(
    {
        "model_name": model_config["name"],
        "num_shots": args.num_shots,
        "global_icu_stay_id": icu_ids,
        "actual_label": actual_labels,
        "predicted_label": predictions,
    }
)
# Set global_icu_stay_id as the index
predictions_df.set_index("global_icu_stay_id", inplace=True)

predictions_df.to_csv(output_csv_path)
print(f"Per-patient evaluation results for {model_config['name']} saved to {output_csv_path}") # fmt: skip

### CALCULATE METRICS ###
# These metrics are for console output / plot titles, not the primary CSV output anymore.
model_metrics = {
    "accuracy": accuracy_score(actual_labels, predictions),
    "auc": roc_auc_score(actual_labels, predictions),
    "f1": f1_score(actual_labels, predictions, zero_division=0),
    "confusion_matrix": confusion_matrix(actual_labels, predictions).tolist(),
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
# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(actual_labels, predictions)
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
