import argparse
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from _utils import bootstrap_auc
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    roc_curve,
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# fmt: off
# Define TabuLa specific prompt components
TABULA_TARGET_COL_NAME = "mortality_in_icu"
QUARTILE_LABELS = ["0.0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1.0"]
TABULA_LABEL_OPTIONS = "||" + "||".join(QUARTILE_LABELS) + "||"
TABULA_PREFIX = f"Predict the value of {TABULA_TARGET_COL_NAME}: {TABULA_LABEL_OPTIONS} "
TABULA_SUFFIX = f" What is the value of {TABULA_TARGET_COL_NAME}? {TABULA_LABEL_OPTIONS}"
# Special tokens for TabuLa
TABULA_END_INPUT_TOKEN = "<|endinput|>"
TABULA_END_COMPLETION_TOKEN = "<|endcompletion|>"
# fmt: on


# ------------------------------------------------------------------------------
# region loading data
def load_serialized_data(file_path):
    """Loads serialized summaries and extracts Global ICU Stay ID."""
    summaries = []
    is_tabula_file = "tabula_serialized_data" in os.path.basename(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if is_tabula_file:
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    icu_id_str, summary_text = parts
                    summaries.append(
                        {
                            "global_icu_stay_id": icu_id_str,
                            "summary_text": summary_text,
                        }
                    )
                else:
                    print(
                        f"Warning: TabuLa line format error (expected ID\\tTEXT): {line}"
                    )
            else:  # Original logic for non-TabuLa files
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
    model_id: str,
    few_shot_prompt_text: str,
    num_shots: int,
    hf_tokenizer=None,
    hf_model=None,
):
    """Processes a single row for LLM evaluation."""
    _, row = row_tuple  # row_tuple is (index, series) from df.iterrows()
    summary_text = row["summary_text"]
    actual_label = row["mortality_in_icu"]
    icu_id = row["global_icu_stay_id"]

    prompt_parts = []
    is_tabula_model = "tabula" in model_name.lower()

    if is_tabula_model:
        # For TabuLa, few_shot_prompt_text is already formatted with prefix/suffix/outcome
        if num_shots > 0 and few_shot_prompt_text:
            prompt_parts.append(few_shot_prompt_text)
        # Append the query for the current instance
        # Format: PREFIX + features + SUFFIX + END_INPUT_TOKEN
        query_prompt_part = f"{TABULA_PREFIX}{summary_text}{TABULA_SUFFIX} {TABULA_END_INPUT_TOKEN}"
        prompt_parts.append(query_prompt_part)
    else:  # non-TabuLa prompt construction
        prompt_parts.append(
            "You are a useful medical assistant. "
            "Your task is to determine whether a patient will die in the ICU based on their summary.\n"
            "You will be given a summary of the patient's ICU stay"
        )
        if num_shots > 0 and few_shot_prompt_text:
            prompt_parts.append(" and some additional examples.\n")
            prompt_parts.append(few_shot_prompt_text)
        else:  # Ensure correct punctuation
            prompt_parts.append(".\n")

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
    answer_part = ""  # Initialize to avoid UnboundLocalError

    try:
        # --- Ollama inference (commented out, using Huggingface instead) ---
        # options = {"num_ctx": 8192, "num_predict": 100}  # Ollama options commented out
        # response = ollama.generate(
        #     model=model_name,
        #     prompt=prompt,
        #     # options=options,
        #     think="think" in model_id,  # Use think mode for better reasoning
        # )
        # answer_part = response["response"].strip().lower()
        # full_response_text = response["response"]
        # --- Huggingface Transformers inference (no pipeline) ---
        model_device = hf_model.device

        # Try to apply chat template, fall back to direct prompt if not available
        try:
            model_prompt = hf_tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking="think" in model_id,
            )
        except ValueError:
            # Model doesn't have a chat template, use prompt directly
            model_prompt = prompt

        model_inputs = hf_tokenizer([model_prompt], return_tensors="pt").to(
            model_device
        )
        generated_ids = hf_model.generate(**model_inputs, max_new_tokens=100)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(
                model_inputs.input_ids, generated_ids
            )
        ]
        answer_part = (
            hf_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
                0
            ]
            .strip()
            .lower()
        )
        full_response_text = answer_part

        if is_tabula_model:
            # Interpret TabuLa quartile prediction and convert to probability
            if QUARTILE_LABELS[0].lower() in answer_part:  # "0.0-0.25"
                predicted_probability = 0.125
            elif QUARTILE_LABELS[1].lower() in answer_part:  # "0.25-0.5"
                predicted_probability = 0.375
            elif QUARTILE_LABELS[2].lower() in answer_part:  # "0.5-0.75"
                predicted_probability = 0.625
            elif QUARTILE_LABELS[3].lower() in answer_part:  # "0.75-1.0"
                predicted_probability = 0.875
            else:
                print(
                    f"Warning: Model {model_name} (TabuLa) produced an unclear quartile answer: "
                    f"'{answer_part}' for summary ID {icu_id}."
                    f"Actual label: {actual_label}\n"
                    f"Full response: {full_response_text}"
                )
        else:
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
            f"Error during model generation for model {model_name}, "
            f"ID {icu_id}: {e}"
        )

    return icu_id, int(actual_label), predicted_probability


def evaluate_model(
    test_df: pd.DataFrame,
    model_name: str,
    model_id: str,
    train_df: pd.DataFrame,
    num_shots: int,
    max_workers: int = 10,
    hf_tokenizer=None,
    hf_model=None,
):
    """Evaluates a given LLM model (via Ollama), optionally with few-shot examples, using parallel processing."""
    print(f"Evaluating {model_name} with {num_shots}-shot prompting...")
    is_tabula_model = "tabula" in model_name.lower()

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
            fs_summary_text = fs_row["summary_text"]
            if is_tabula_model:
                fs_outcome_val = fs_row["mortality_in_icu"]
                # Map binary outcome to extreme quartiles for TabuLa few-shot
                fs_outcome_quartile = (
                    QUARTILE_LABELS[3]
                    if int(fs_outcome_val) == 1
                    else QUARTILE_LABELS[0]
                )
                # Format the few-shot example for TabuLa
                formatted_fs_example = (
                    TABULA_PREFIX
                    + fs_summary_text
                    + TABULA_SUFFIX
                    + TABULA_END_INPUT_TOKEN
                    + fs_outcome_quartile
                    + TABULA_END_COMPLETION_TOKEN
                    + "\n"
                )
                temp_prompt_parts.append(formatted_fs_example)
            else:  # Original behavior for non-TabuLa: just append the summary text
                temp_prompt_parts.append(fs_summary_text)
        few_shot_prompt_text = "".join(temp_prompt_parts)

    # Limit test_df if using think mode to avoid excessive processing
    if "think" in model_id:
        test_df = test_df.head(100)
        print(f"Limited test set to {len(test_df)} samples for think mode")

    results_list = []
    tasks = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for row_tuple in test_df.iterrows():
            tasks.append(
                executor.submit(
                    _process_row,
                    row_tuple,
                    model_name,
                    model_id,
                    few_shot_prompt_text,
                    num_shots,
                    hf_tokenizer,
                    hf_model,
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
    "--condition",
    type=str,
    default="basic",
    help="Condition identifier for output filenames (e.g., 'basic', 'hourly', 'forward_fill').",
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
    args.output_dir,
    f"llm_{args.model}_{args.condition}_{args.num_shots}-shot_results.csv",
)
plot_output_path = os.path.join(
    args.plot_dir,
    f"llm_{args.model}_{args.condition}_{args.num_shots}-shot_roc_curve.png",
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
    # {"name": "llama3.2:3b-instruct",         "id": "llama_3.2_3b"},
    # {"name": "llama3.1:8b-instruct",         "id": "llama_3.1_8b"},
    # {"name": "qwen3:8b",                     "id": "qwen_3_8b"},
    {"name": "Qwen/Qwen3-8B",                  "id": "qwen_3_8b"},
    # {"name": "qwen3:8b",                     "id": "qwen_3_8b_think"},
    # {"name": "granite3.3:8b-instruct",       "id": "granite_3.3_8b"},
    # {"name": "mistral:7b-instruct-v0.3",     "id": "mistral_7b"},
    # {"name": "deepseek-r1:8b-llama-distill", "id": "deepseek_r1_8b"},
    # {"name": "gemma3:4b-it",                 "id": "gemma_3_4b"},
    # {"name": "medgemma3:4b-it",              "id": "medgemma_3_4b"}, # https://huggingface.co/unsloth/medgemma-4b-it-GGUF
    {"name": "google/medgemma-4b-it",          "id": "medgemma_3_4b"},
    # {"name": "tabula:8b",                    "id": "tabula_8b"},     # https://huggingface.co/tensorblock/tabula-8b-GGUF
    {"name": "mlfoundations/tabula-8b",        "id": "tabula_8b"}      # https://huggingface.co/mlfoundations/tabula-8b
] # fmt: skip

# Select the target model
model_config = None
for m_config in model_configs:
    if m_config["id"] == args.model:
        model_config = m_config
        break

model_results = {}

################################################################################
# Load Huggingface Transformers model and tokenizer
print(f"Loading Huggingface model and tokenizer for {model_config['name']} ...")
hf_tokenizer = AutoTokenizer.from_pretrained(model_config["name"])

# Set pad_token to avoid warnings during generation
if hf_tokenizer.pad_token is None:
    hf_tokenizer.pad_token = hf_tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)
hf_model = AutoModelForCausalLM.from_pretrained(
    model_config["name"],
    torch_dtype="auto",
    device_map="auto",
    quantization_config=bnb_config,
    max_memory={0: "38GiB"},
)

################################################################################
# EVALUATE LLM
print(f"--- Starting evaluation for {model_config['name']} ---")

icu_ids, actual_labels, probability_predictions = evaluate_model(
    test_df=test_df,
    model_name=model_config["name"],
    model_id=model_config["id"],
    train_df=train_df,
    num_shots=args.num_shots,
    hf_tokenizer=hf_tokenizer,
    hf_model=hf_model,
)

# Unload the model to free up resources
# ollama.generate(
#     model=model_config["name"],
#     prompt="Done evaluating.",
#     options={"num_predict": 1},
#     keep_alive=0,
# )

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

### CALCULATE METRICS ###

# Calculate AUC with bootstrapping
try:
    auc_stats = bootstrap_auc(
        final_actual_labels,
        final_probability_predictions,
        n_bootstrap=100,
        random_state=SEED,
    )
except Exception as e:
    print(f"Skipping AUC bootstrap due to error: {e}")
    auc_stats = {
        "auc": float("nan"),
        "auc_ci_lower": float("nan"),
        "auc_ci_upper": float("nan"),
    }

# For metrics requiring binary predictions, apply a 0.5 threshold

if len(final_probability_predictions) > 0:
    binary_predictions = [
        1 if p >= 0.5 else 0 for p in final_probability_predictions
    ]
else:
    binary_predictions = []

# Calculate metrics only if there's valid data after filtering
try:
    model_metrics = {
        "accuracy": accuracy_score(final_actual_labels, binary_predictions),
        "auc": auc_stats["auc"],
        "auc_ci_lower": auc_stats["auc_ci_lower"],
        "auc_ci_upper": auc_stats["auc_ci_upper"],
        "f1": f1_score(
            final_actual_labels, binary_predictions, zero_division=0
        ),
        "cm": (
            confusion_matrix(final_actual_labels, binary_predictions).tolist()
        ),
        "missingness": none_predictions_count / total_predictions,
    }
except Exception as e:
    print(f"Skipping metric calculation due to error: {e}")
    model_metrics = {
        "accuracy": float("nan"),
        "auc": auc_stats["auc"],
        "auc_ci_lower": auc_stats["auc_ci_lower"],
        "auc_ci_upper": auc_stats["auc_ci_upper"],
        "f1": float("nan"),
        "cm": [],
        "missingness": none_predictions_count / total_predictions,
    }
print(
    f"Model: {model_config['name']}, "
    f"Num Shots: {args.num_shots}, "
    f"missing Predictions: {model_metrics['missingness']:.1%}, "
    f"Accuracy: {model_metrics['accuracy']:.2f}, "
    f"AUC: {model_metrics['auc']:.3f} [{model_metrics['auc_ci_lower']:.3f}-{model_metrics['auc_ci_upper']:.3f}], "
    f"F1 Score: {model_metrics['f1']:.2f}, "
    f"Confusion Matrix: {model_metrics['cm']}"
)

### SAVE RESULTS ###
# Create a DataFrame with patient IDs, actual labels, and predicted probabilities
try:
    predictions_df = pd.DataFrame(
        {
            "model_name": model_config["name"],
            "model_args": f"{args.model}-{args.condition}",
            "num_shots": args.num_shots,
            "auc_ci_lower": auc_stats["auc_ci_lower"],
            "auc_ci_upper": auc_stats["auc_ci_upper"],
            "global_icu_stay_id": icu_ids,
            "actual_label": actual_labels,
            "predicted_probability": probability_predictions,
        }
    )
    predictions_df.set_index("global_icu_stay_id", inplace=True)
    predictions_df.to_csv(output_csv_path)
    print(
        f"Per-patient evaluation results for {model_config['name']} saved to {output_csv_path}"
    )
except Exception as e:
    print(f"Skipping saving predictions due to error: {e}")

### SAVE ROC CURVE ###
# Calculate ROC curve using filtered probabilities
try:
    fpr, tpr, thresholds = roc_curve(
        final_actual_labels, final_probability_predictions
    )
    roc_auc_val = auc(fpr, tpr)

    plt.figure()
    plt.plot(
        fpr,
        tpr,
        lw=2,
        label=f"ROC curve (AUC = {roc_auc_val:.2f} [{auc_stats['auc_ci_lower']:.2f}-{auc_stats['auc_ci_upper']:.2f}])",
    )
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
except Exception as e:
    print(f"Skipping ROC curve plot due to error: {e}")
