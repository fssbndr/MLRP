import glob
import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_dir", required=True, help="Input directory containing result CSVs"
)
parser.add_argument(
    "--output_csv", required=True, help="Output CSV file for aggregation"
)
args = parser.parse_args()

INPUT_DIR = os.path.abspath(args.input_dir)
OUTPUT_FILE = os.path.abspath(args.output_csv)

# Patterns to match
LLM_PATTERN = os.path.join(INPUT_DIR, "llm*shot_results.csv")
TABPFN_PATTERN = os.path.join(INPUT_DIR, "tabpfn*shot_results.csv")
BASELINE_PATTERN = os.path.join(INPUT_DIR, "baseline*results.csv")

### Wildcard parameter sets from Snakefile
LLM_MODEL_IDS = ["qwen_3_8b", "tabula_8b", "medgemma_3_4b"]
CONDITIONS = ["basic", "hourly", "forward_fill", "hourly_stats", "minmax", "stats"]
NUM_SHOT_VALUES = [0] + [2**i for i in range(6)]
NUM_SHOT_VALUES_TABPFN = [2**i for i in range(6)]


# Collect files via glob
llm_files = glob.glob(LLM_PATTERN)
tabpfn_files = glob.glob(TABPFN_PATTERN)
baseline_files = glob.glob(BASELINE_PATTERN)
print(
    f"Found {len(llm_files)} LLM files, "
    f"{len(tabpfn_files)} TabPFN files, "
    f"and {len(baseline_files)} baseline files to aggregate."
)
all_files = llm_files + tabpfn_files + baseline_files
print(f"Total candidate files: {len(all_files)}")

records = []
for fp in sorted(all_files):
    if not os.path.exists(fp) or os.path.getsize(fp) == 0:
        continue
    df = pd.read_csv(fp)
    
    basename = os.path.basename(fp)
    name_no_suffix = basename.replace("_results.csv", "")
    # Determine model_args by matching known wildcards
    if name_no_suffix.startswith("llm_"):
        matched = False
        for model_id in LLM_MODEL_IDS:
            prefix = f"llm_{model_id}_"
            if name_no_suffix.startswith(prefix):
                tail = name_no_suffix[len(prefix):]  # e.g. 'forward_fill_128-shot'
                if "_" in tail:
                    # Split at the last underscore to allow any condition and any shot count
                    last_underscore = tail.rfind("_")
                    if last_underscore != -1:
                        cond = tail[:last_underscore]
                        shots_part = tail[last_underscore + 1 :]
                        if shots_part.endswith("-shot"):
                            df["model_args"] = f"{cond}"
                            matched = True
                            break
        if not matched:
            print(f"Skipping unmatched LLM file: {basename}")
            continue
    elif name_no_suffix.startswith("tabpfn"):
        # TabPFN: remove prefix and keep condition-coded names
        df["model_args"] = (
            df["model_args"]
            .str.replace("tabpfn_", "")
            .str.replace("tabpfn", "basic")
            .str.replace("-", "_")
        )
    elif name_no_suffix.startswith("baseline_"):
        # For baseline files, set model_args to 'baseline_{condition}' where condition is in CONDITIONS
        found_condition = None
        for cond in CONDITIONS:
            if name_no_suffix.endswith(f"_{cond}"):
                found_condition = cond.replace("-", "_")
                break
        if found_condition:
            df["model_args"] = f"baseline_{found_condition}"
        else:
            # fallback: just use the file stem
            df["model_args"] = f"baseline_basic"
    else:
        # Skip unknown patterns
        continue

    records.append(df)

agg = pd.concat(records, ignore_index=True)
agg.to_csv(OUTPUT_FILE, index=False)
print(f"Aggregated {len(records)} files into {OUTPUT_FILE}.")