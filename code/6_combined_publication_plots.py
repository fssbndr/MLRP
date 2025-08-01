# --- Combined Publication Plots Script ---
import argparse
import os
import warnings
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import bootstrap
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Set publication-ready style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

parser = argparse.ArgumentParser(
    description="Combined evaluation and publication plots script"
)
parser.add_argument(
    "--input_csv",
    required=True,
    help="Path to evaluation_results.csv",
)
parser.add_argument(
    "--output_dir",
    required=True,
    help="Output directory for plots",
)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# Load data
df = pd.read_csv(args.input_csv)
print(f"Loaded {len(df)} records from input CSV")


# Helper to shorten model_name for display/legend
def shorten_model_name(model_name):
    model_name = str(model_name).lower()
    if "logistic" in model_name:
        return "Logistic"
    elif "xgboost" in model_name:
        return "XGBoost"
    elif "catboost" in model_name:
        return "CatBoost"
    elif "lightgbm" in model_name:
        return "LightGBM"
    elif "tabpfn" in model_name:
        return "TabPFN"
    elif "kaplanmeier" in model_name:
        return "Kaplan-Meier"
    elif "cox" in model_name:
        return "Cox PH"
    elif "qwen" in model_name:
        return "Qwen"
    elif "tabula" in model_name:
        return "TabuLa"
    elif "medgemma" in model_name:
        return "MedGemma"
    else:
        return model_name


# Helper to extract condition from model_args
def extract_condition(model_args):
    model_args = str(model_args).lower()
    if "hourly_stats" in model_args:
        return "Hourly + Stats"
    elif "forward_fill" in model_args:
        return "Forward Fill"
    elif "hourly" in model_args:
        return "Hourly"
    elif "stats" in model_args:
        return "Statistics"
    elif "minmax" in model_args:
        return "Min-Max"
    elif "basic" in model_args:
        return "Basic"
    else:
        return "Other"


# No longer add model_type to dataframe; use model_name and model_args directly

# Define consistent colors
model_colors = {
    "Logistic": "#1f77b4",
    "XGBoost": "#ff7f0e",
    "TabPFN": "#2ca02c",
    "Kaplan-Meier": "#d62728",
    "Cox PH": "#9467bd",
    "Qwen": "#e377c2",
    "TabuLa": "#17becf",
    "MedGemma": "#bcbd22",
    "LLM": "#7f7f7f",
    "Other": "#8c564b",
}
condition_colors = {
    "Basic": "#1f77b4",
    "Hourly": "#ff7f0e",
    "Statistics": "#2ca02c",
    "Min-Max": "#d62728",
    "Forward Fill": "#9467bd",
    "Hourly + Stats": "#8c564b",
}


def add_oasis_curve(ax):
    # Try to find OASIS row in the input DataFrame (not summary)
    oasis_row = df[
        (df["model_name"].str.contains("oasis", case=False, na=False))
    ]

    y_true_oasis = oasis_row["actual_label"].values
    y_pred_oasis = oasis_row["predicted_probability"].values
    mask_oasis = ~((pd.isna(y_true_oasis)) | (pd.isna(y_pred_oasis)))
    y_true_oasis = y_true_oasis[mask_oasis]
    y_pred_oasis = y_pred_oasis[mask_oasis]
    fpr_oasis, tpr_oasis, _ = roc_curve(y_true_oasis, y_pred_oasis)
    ax.plot(
        fpr_oasis,
        tpr_oasis,
        label=f"OASIS (AUC={oasis_auc:.3f})",
        color="grey",
        linestyle=":",
        linewidth=2,
        alpha=0.8,
    )


# --- FIGURE 1: ROC curves for different conditions for each model (roc_conditions_by_model.png) ---
def plot_roc_conditions_by_model(df, output_path, oasis_auc=None):
    # Only include baselines (model_args contains 'baseline'), exclude Kaplan-Meier
    df_baseline = df[
        (df["model_args"].str.contains("baseline", case=False, na=False))
        & (df["model_name"].apply(shorten_model_name) != "Kaplan-Meier")
        & (df["model_name"].apply(shorten_model_name) != "oasis")
    ]

    model_names = sorted(
        df_baseline["model_name"].unique(), key=shorten_model_name
    )
    n_models = len(model_names)
    n_cols = 2
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6 * n_rows))
    axes = axes.flatten()
    for i, model_name in enumerate(model_names):
        ax = axes[i]
        model_data = df_baseline[df_baseline["model_name"] == model_name]
        conditions = sorted(
            model_data["model_args"].unique(), key=extract_condition
        )
        for condition in conditions:
            condition_data = model_data[model_data["model_args"] == condition]
            y_true = condition_data["actual_label"].values
            y_pred = condition_data["predicted_probability"].values
            mask = ~((pd.isna(y_true)) | (pd.isna(y_pred)))
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            if len(np.unique(y_true)) > 1 and len(y_true) > 0:
                fpr, tpr, _ = roc_curve(y_true, y_pred)
                auc_val = roc_auc_score(y_true, y_pred)
                cond_label = extract_condition(condition)
                ax.plot(
                    fpr,
                    tpr,
                    label=f"{cond_label} (AUC={auc_val:.2f})",
                    color=condition_colors.get(cond_label, None),
                )
        # Plot OASIS as a grey dotted ROC curve if available
        if oasis_auc is not None:
            add_oasis_curve(ax)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(
            f"{shorten_model_name(model_name)} - Different Conditions",
            fontweight="bold",
            fontsize=14,
        )
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


# --- FIGURE 2: ROC curves for different models for each condition (roc_models_by_condition.png) ---
def plot_roc_models_by_condition(df, output_path, oasis_auc=None):
    # Only include baselines (model_args contains 'baseline'), exclude Kaplan-Meier
    df_baseline = df[
        (df["model_args"].str.contains("baseline", case=False, na=False))
        & (df["model_name"].apply(shorten_model_name) != "Kaplan-Meier")
    ]
    # Restrict to results with the most amounts of shots per model
    idx = (
        df_baseline.groupby("model_name")["num_shots"].transform("max")
        == df_baseline["num_shots"]
    )
    df_maxshots = df_baseline[idx].copy()
    all_conditions = [
        "baseline_" + x
        for x in [
            "basic",
            "minmax",
            "stats",
            "hourly",
            "forward_fill",
            "hourly_stats",
        ]
    ]
    n_conditions = len(all_conditions)
    n_cols = 2
    n_rows = (n_conditions + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6 * n_rows))
    axes = axes.flatten()
    for i, condition in enumerate(all_conditions):
        print(f"Processing condition: {condition}")
        print(
            f"Number of models in condition: {len(df_maxshots[df_maxshots['model_args'] == condition])}"
        )

        ax = axes[i]
        condition_data = df_maxshots[df_maxshots["model_args"] == condition]
        model_names_in_condition = sorted(
            condition_data["model_name"].unique(), key=shorten_model_name
        )
        for model_name in model_names_in_condition:
            print(f"Processing model: {model_name}")
            print(
                f"Number of records for model: {len(condition_data[condition_data['model_name'] == model_name])}"
            )
            model_data = condition_data[
                condition_data["model_name"] == model_name
            ]
            y_true = model_data["actual_label"].values
            y_pred = model_data["predicted_probability"].values
            mask = ~((pd.isna(y_true)) | (pd.isna(y_pred)))
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            if len(np.unique(y_true)) > 1 and len(y_true) > 0:
                fpr, tpr, _ = roc_curve(y_true, y_pred)
                auc_val = roc_auc_score(y_true, y_pred)
                ax.plot(
                    fpr,
                    tpr,
                    label=f"{shorten_model_name(model_name)} (AUC={auc_val:.2f})",
                    color=model_colors.get(
                        shorten_model_name(model_name), None
                    ),
                )
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
        # Plot OASIS as a grey dotted ROC curve if available
        if oasis_auc is not None:
            add_oasis_curve(ax)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(
            f"{extract_condition(condition)} Condition - Different Models",
            fontweight="bold",
            fontsize=14,
        )
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


# --- Compute summary metrics (with bootstrapped AUC CIs) ---
def compute_summary_metrics(df_raw):
    df = df_raw.copy()
    df["actual_label"] = pd.to_numeric(df["actual_label"], errors="coerce")
    df["predicted_probability"] = pd.to_numeric(
        df["predicted_probability"], errors="coerce"
    )
    df.dropna(subset=["actual_label", "predicted_probability"], inplace=True)
    results = []
    for name, group in df.groupby(["model_name", "model_args", "num_shots"]):
        y_true = group["actual_label"].values
        y_pred = group["predicted_probability"].values
        if len(y_true) < 2 or len(np.unique(y_true)) < 2:
            continue
        auc_val = roc_auc_score(y_true, y_pred)
        ap_val = average_precision_score(y_true, y_pred)
        indices = np.arange(len(y_true))

        def stat(idx):
            return roc_auc_score(y_true[idx], y_pred[idx])

        try:
            res = bootstrap(
                (indices,),
                stat,
                confidence_level=0.95,
                method="percentile",
                n_resamples=1000,
                random_state=42,
            )
            ci_low = res.confidence_interval.low
            ci_high = res.confidence_interval.high
        except:
            ci_low, ci_high = np.nan, np.nan
        y_pred_bin = (y_pred >= 0.5).astype(int)
        metrics = {
            "model_name": shorten_model_name(name[0]),
            "model_args": name[1] if isinstance(name, tuple) else name[0],
            "num_shots": name[2] if isinstance(name, tuple) else None,
            "AUC": auc_val,
            "AUC_CI_Low": ci_low,
            "AUC_CI_High": ci_high,
            "AP": ap_val,
            "Accuracy": accuracy_score(y_true, y_pred_bin),
            "F1 Score": f1_score(y_true, y_pred_bin, zero_division=0),
            "Precision": precision_score(y_true, y_pred_bin, zero_division=0),
            "Recall": recall_score(y_true, y_pred_bin, zero_division=0),
        }
        results.append(metrics)
    df_metrics = pd.DataFrame(results)
    return df_metrics


df_metrics = compute_summary_metrics(df)

# Save summary metrics
df_metrics.to_csv(
    os.path.join(args.output_dir, "summary_metrics.csv"),
    index=False,
    float_format="%.3f",
)
with open(os.path.join(args.output_dir, "summary_metrics.md"), "w") as f:
    f.write(df_metrics.to_markdown(index=False, floatfmt=".3f"))


# --- Metrics evolution with shots (metrics_evolution_with_shots.png) ---
def plot_metrics_evolution(df, output_path, oasis_metrics=None):
    """Generates and saves line plots of performance metrics vs. 'num_shots'.
    If oasis_metrics is provided, adds horizontal lines for OASIS metrics.
    """
    metrics = ["AUC", "Accuracy", "F1 Score", "Precision", "Recall"]
    df_plot = df.copy().sort_values(by=["model_name", "num_shots"])
    df_melted = df_plot.melt(
        id_vars=["model_name", "num_shots", "model_args"],
        value_vars=[m for m in metrics if m in df_plot.columns],
        var_name="Metric",
        value_name="Score",
    )
    g = sns.relplot(
        data=df_melted,
        x="num_shots",
        y="Score",
        hue="model_args",
        col="Metric",
        kind="line",
        marker="o",
        height=4,
        aspect=1.2,
        palette="tab10",
        facet_kws={"sharey": False, "sharex": True},
        legend=True,
    )
    g.fig.suptitle("Metrics vs. Number of Shots by Model", y=1.03)
    g.set_axis_labels("Number of Shots", "Score")
    for ax, metric in zip(g.axes.flat, metrics):
        ax.set_ylim(0, 1)
        ax.set_xlim(left=0, right=128)
        ax.set_xscale("symlog", base=2)
        # Add OASIS horizontal line if available
        if oasis_metrics is not None and metric in oasis_metrics:
            ax.axhline(
                oasis_metrics[metric],
                color="grey",
                linestyle=":",
                linewidth=2,
                label=f"OASIS ({metric}={oasis_metrics[metric]:.3f})",
                alpha=0.8,
            )
        if ax.has_data():
            handles, labels = ax.get_legend_handles_labels()
            if handles and labels:
                unique_entries = {}
                for handle, label in zip(handles, labels):
                    if label not in unique_entries and not label.startswith(
                        "_"
                    ):
                        unique_entries[label] = handle
                if unique_entries:
                    ax.legend(
                        unique_entries.values(),
                        unique_entries.keys(),
                        title="model_name",
                        loc="best",
                        fontsize="small",
                    )
    if hasattr(g, "_legend") and g._legend is not None:
        g._legend.remove()
    g.fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path)
    plt.close(g.fig)
    print(f"Saved: {output_path}")


# --- AUC evolution vs. #shots for basic condition (with OASIS AUC line) ---
def plot_auc_vs_shots(df_metrics, output_path, oasis_auc=None):
    df_sm = df_metrics.copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    models_basic = sorted(df_sm["model_name"].unique(), key=shorten_model_name)
    colors_basic = {
        m: model_colors.get(shorten_model_name(m), "#333333")
        for m in models_basic
    }
    for m in models_basic:
        print(f"Processing model: {m}")
        sub = df_sm[df_sm["model_name"] == m].sort_values("num_shots")
        shots = sub["num_shots"]
        mean_auc = sub["AUC"]
        ax.plot(
            shots,
            mean_auc,
            label=shorten_model_name(m),
            color=colors_basic[m],
            linewidth=2,
            marker="o",
        )
    if oasis_auc is not None:
        ax.axhline(
            oasis_auc,
            color="grey",
            linestyle=":",
            linewidth=2,
            label=f"OASIS (AUC={oasis_auc:.3f})",
        )
    ax.set_xscale("symlog", base=2)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Number of Shots", fontsize=12)
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_title(
        f"AUC Evolution for Condition {df_metrics['model_args'].iloc[0]}",
        fontweight="bold",
        fontsize=14,
    )
    ax.grid(alpha=0.3)
    ax.legend(title="Model", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


# --- Generate LaTeX table for LLMs and TabPFN (llm_tabpfn_full_results style) ---
def generate_shot_results_tables(df_metrics, output_dir):
    """
    Generate LaTeX tables for shot_results (basic) and shot_results_stats (stats) conditions,
    matching the format in report.tex for Table~\ref{tab:shot_results} and Table~\ref{tab:shot_results_stats}.
    """
    models = ["Qwen", "TabuLa", "MedGemma", "TabPFN"]
    shots = [0, 1, 2, 4, 8, 16, 32, 64, 128]
    conditions = {"shot_results": "basic", "shot_results_stats": "stats"}
    for table_name, cond in conditions.items():
        lines = []
        lines.append("\\begin{table}[!htb]")
        lines.append("  \\centering")
        lines.append("  \\small")
        lines.append(
            f"  \\caption{{Effect of shot count on ROC-AUC (AUC [95\\% CI]) for TabPFN and LLMs in the {cond} condition.}}"
        )
        lines.append(f"  \\label{{tab:{table_name}}}")
        lines.append("  \\begin{tabular}{lcccc}")
        lines.append("  \\toprule")
        lines.append(
            "  Shots   & Qwen & TabuLa & MedGemma & TabPFN             \\\\"
        )
        lines.append("  \\midrule")
        for shot in shots:
            row = []
            shot_label = f"{shot}-shot"
            row.append(shot_label)
            for model in models:
                mask = (
                    (df_metrics["model_name"].str.lower() == model.lower())
                    & (df_metrics["model_args"] == cond)
                    & (df_metrics["num_shots"] == shot)
                )
                if mask.any():
                    r = df_metrics[mask].iloc[0]
                    auc = r["AUC"]
                    ci_low = r["AUC_CI_Low"]
                    ci_high = r["AUC_CI_High"]
                    if (
                        pd.notnull(auc)
                        and pd.notnull(ci_low)
                        and pd.notnull(ci_high)
                    ):
                        cell = f"{auc:.3f} [{ci_low:.3f}--{ci_high:.3f}]"
                        # Bold best AUC in row
                        row.append(cell)
                    else:
                        row.append("--")
                else:
                    row.append("--")
            # Bold the best AUC in the row (ignoring '--')
            auc_vals = [
                (float(re.match(r"([0-9.]+)", c).group(1)), i)
                for i, c in enumerate(row[1:])
                if c != "--"
            ]
            if auc_vals:
                max_auc = max(auc_vals, key=lambda x: x[0])[0]
                for val, idx in auc_vals:
                    if abs(val - max_auc) < 1e-8:
                        row[idx + 1] = f"\\textbf{{{row[idx+1]}}}"
            lines.append("  " + " & ".join(row) + " \\\\")
        lines.append("  \\bottomrule")
        lines.append("  \\end{tabular}")
        lines.append("\\end{table}")
        with open(os.path.join(output_dir, f"{table_name}.tex"), "w") as f:
            f.write("\n".join(lines))
        print(
            f"Saved LaTeX table to {os.path.join(output_dir, f'{table_name}.tex')}"
        )


def generate_llm_tabpfn_full_results_table(df_metrics, output_path):
    """
    Generate a LaTeX longtable for LLMs (Qwen, TabuLa, MedGemma) and TabPFN across all conditions and shot counts,
    matching the format of the llm_tabpfn_full_results table in the report.
    """
    # Define models and conditions of interest
    models = ["Qwen", "TabuLa", "MedGemma", "TabPFN"]
    conditions = [
        ("Basic", [0, 1, 2, 4, 8, 16, 32, 64, 128]),
        ("Min-Max", [0, 1, 2, 4, 8, 16, 32, 64, 128]),
        ("Statistics", [0, 1, 2, 4, 8, 16, 32, 64, 128]),
        ("Hourly", [0, 1, 2, 4, 8, 16, 32, 64, 128]),
        ("Hourly+Stats", [0, 1, 2, 4, 8, 16, 32, 64, 128]),
        ("Forward Fill", [0, 1, 2, 4, 8, 16, 32, 64, 128]),
    ]
    # Map condition names to model_args substrings
    cond_map = {
        "Basic": "basic",
        "Min-Max": "minmax",
        "Statistics": "stats",
        "Hourly": "hourly",
        "Hourly+Stats": "hourly_stats",
        "Forward Fill": "forward_fill",
    }
    # Start LaTeX table
    lines = []
    lines.append("\\begin{longtable}{lcccc}")
    lines.append(
        "  \\caption{ROC-AUC [95\\% CI] for LLMs (Qwen, TabuLa, MedGemma) and TabPFN across main feature engineering strategies and shot counts. Dashes (--) indicate not evaluated.}"
    )
    lines.append("  \\label{tab:llm_tabpfn_full_results} \\\\")
    lines.append("    \\toprule")
    lines.append("    Shot & Qwen & TabuLa & MedGemma & TabPFN \\\\")
    lines.append("    \\midrule")
    lines.append("    \\endfirsthead\n")
    lines.append("    \\multicolumn{5}{c}%")
    lines.append(
        "    {{\\bfseries \\tablename\\ \\thetable{} -- continued from previous page}} \\\\"
    )
    lines.append("    \\toprule")
    lines.append("    & Qwen & TabuLa & MedGemma & TabPFN \\\\")
    lines.append("    \\midrule")
    lines.append("    \\endhead\n")
    lines.append(
        "    \\midrule \\multicolumn{5}{r}{{Continued on next page}} \\\\"
    )
    lines.append("    \\endfoot\n")
    lines.append("    \\bottomrule")
    lines.append("    \\endlastfoot\n")
    for cond, shots in conditions:
        lines.append(f"    \\multicolumn{{5}}{{c}}{{\\textbf{{{cond}}}}} \\\\")
        lines.append("    \\midrule")
        for shot in shots:
            row = [f"{shot}-shot"]
            auc_values = []
            cell_strings = []
            for model in models:
                mask = (
                    (df_metrics["model_name"].str.lower() == model.lower())
                    & (
                        df_metrics["model_args"].str.contains(
                            cond_map[cond], case=False, na=False
                        )
                    )
                    & (df_metrics["num_shots"] == shot)
                )
                if mask.any():
                    r = df_metrics[mask].iloc[0]
                    auc = r["AUC"]
                    ci_low = r["AUC_CI_Low"]
                    ci_high = r["AUC_CI_High"]
                    if (
                        pd.notnull(auc)
                        and pd.notnull(ci_low)
                        and pd.notnull(ci_high)
                    ):
                        cell = f"{auc:.3f} [{ci_low:.3f}--{ci_high:.3f}]"
                        auc_values.append((auc, len(cell_strings)))
                    else:
                        cell = "--"
                        auc_values.append((None, len(cell_strings)))
                else:
                    cell = "--"
                    auc_values.append((None, len(cell_strings)))
                cell_strings.append(cell)
            # Find the index(es) of the max AUC (ignore None)
            auc_only = [(v, idx) for v, idx in auc_values if v is not None]
            if auc_only:
                max_auc = max(auc_only, key=lambda x: x[0])[0]
                for i, (v, idx) in enumerate(auc_values):
                    if v is not None and abs(v - max_auc) < 1e-8:
                        # Bold the cell
                        cell_strings[idx] = f"\\textbf{{{cell_strings[idx]}}}"
            row.extend(cell_strings)
            lines.append("    " + " & ".join(row) + " \\\\")
        lines.append("\n    \\addlinespace[1em]")
    lines.append("\\end{longtable}\n")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved LaTeX table to {output_path}")


# --- Main block: call all required plotting functions ---
if __name__ == "__main__" or True:
    oasis_row = df_metrics[
        df_metrics["model_name"].str.contains("oasis", case=False, na=False)
    ]
    oasis_auc = oasis_row.iloc[0]["AUC"]
    oasis_metrics = {
        "AUC": oasis_row.iloc[0]["AUC"],
        "Accuracy": oasis_row.iloc[0]["Accuracy"],
        "F1 Score": oasis_row.iloc[0]["F1 Score"],
        "Precision": oasis_row.iloc[0]["Precision"],
        "Recall": oasis_row.iloc[0]["Recall"],
    }

    # Figure 1: ROC curves for different conditions for each model
    plot_roc_conditions_by_model(
        df,
        os.path.join(args.output_dir, "roc_conditions_by_model.png"),
        oasis_auc=oasis_auc,
    )
    # Figure 2: ROC curves for different models for each condition
    plot_roc_models_by_condition(
        df,
        os.path.join(args.output_dir, "roc_models_by_condition.png"),
        oasis_auc=oasis_auc,
    )
    # Metrics evolution with shots
    plot_metrics_evolution(
        df_metrics,
        os.path.join(args.output_dir, "metrics_evolution_with_shots.png"),
        oasis_metrics=oasis_metrics,
    )
    # Per-model metrics evolution (for foundation models)
    foundation_models = ["TabPFN", "Qwen", "TabuLa", "MedGemma"]
    conditions = [
        "basic",
        "minmax",
        "stats",
        "hourly",
        "forward_fill",
        "hourly_stats",
    ]

    foundation_df = df_metrics[
        df_metrics["model_name"]
        .str.lower()
        .str.match("|".join(foundation_models).lower() + "|oasis")
        & ~df_metrics["model_args"].str.contains("baseline")
    ]
    for model in foundation_models:
        model_df = foundation_df[
            foundation_df["model_name"]
            .str.lower()
            .str.contains(model.lower(), na=False)
        ]
        plot_metrics_evolution(
            model_df,
            os.path.join(
                args.output_dir,
                f"{model}_metrics_evolution_with_shots.png",
            ),
            oasis_metrics=oasis_metrics,
        )
    # AUC vs. shots for all conditions (with OASIS line)
    for condition in conditions:
        plot_auc_vs_shots(
            foundation_df[df_metrics["model_args"] == condition],
            os.path.join(args.output_dir, f"auc_vs_shots_{condition}.png"),
            oasis_auc=oasis_auc,
        )
    # Generate LaTeX table for LLMs and TabPFN
    generate_llm_tabpfn_full_results_table(
        df_metrics,
        os.path.join(args.output_dir, "llm_tabpfn_full_results.tex"),
    )
    # Generate shot_results and shot_results_stats tables
    generate_shot_results_tables(df_metrics, args.output_dir)
    print(f"All plots and summary metrics saved to {args.output_dir}")
