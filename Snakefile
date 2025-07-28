import os

configfile: "config.yaml"

# Model and condition definitions
LLM_MODEL_IDS = ["qwen_3_8b", "medgemma_3_4b", "tabula_8b"]
# LLM_MODEL_IDS = ["qwen_3_8b", "medgemma_3_4b"]
LLM_MODEL_IDS_NON_TABULA = ["qwen_3_8b", "medgemma_3_4b"]
LLM_MODEL_ID_TABULA = ["tabula_8b"]

NUM_SHOT_VALUES_NON_TABULA = [0] + [2**i for i in range(8)] # up to 128 shots
NUM_SHOT_VALUES_TABULA = [0] + [2**i for i in range(6)] # up to 32 shots
NUM_SHOT_VALUES_TABPFN = [2**i for i in range(8)]

# Conditions for data processing
CONDITIONS = ["basic", "hourly", "forward_fill", "stats", "minmax", "hourly_stats"]

# Baseline model types
BASELINE_MODELS = ["logistic", "xgboost", "cox", "lightgbm", "catboost"]

wildcard_constraints:
    model="|".join(LLM_MODEL_IDS),
    baseline="|".join(BASELINE_MODELS),
    condition="|".join(CONDITIONS),
    shots=r"\d+"

rule all:
    input:
        processed_data=expand(
            "output_data/processed_data_{condition}.parquet",
            condition=CONDITIONS
        ),
        serialized_data=expand(
            "output_data/serialized_data_{condition}.txt",
            condition=CONDITIONS
        ),
        serialized_data_test=expand(
            "output_data/serialized_data_{condition}_test.txt",
            condition=CONDITIONS
        ),
        baseline_plots=expand(
            "output_plots/baseline_{baseline}_{condition}_roc_curve.png",
            baseline=BASELINE_MODELS, condition=CONDITIONS
        ) + expand(
            "output_plots/baseline_tabpfn_{condition}_roc_curve.png",
            condition=CONDITIONS
        ) + ["output_plots/baseline_oasis_basic_roc_curve.png"],
        baseline_results=expand(
            "output_data/baseline_{baseline}_{condition}_results.csv",
            baseline=BASELINE_MODELS, condition=CONDITIONS
        ) + expand(
            "output_data/baseline_tabpfn_{condition}_results.csv",
            condition=CONDITIONS
        ) + ["output_data/baseline_oasis_basic_results.csv"],
        llm_plots=expand(
            "output_plots/llm_{model}_{condition}_{shots}-shot_roc_curve.png",
            model=LLM_MODEL_IDS_NON_TABULA, condition=CONDITIONS, shots=NUM_SHOT_VALUES_NON_TABULA
        ) + expand(
            "output_plots/llm_{model}_{condition}_{shots}-shot_roc_curve.png",
            model=LLM_MODEL_ID_TABULA, condition=CONDITIONS, shots=NUM_SHOT_VALUES_TABULA
        ),
        llm_results=expand(
            "output_data/llm_{model}_{condition}_{shots}-shot_results.csv",
            model=LLM_MODEL_IDS_NON_TABULA, condition=CONDITIONS, shots=NUM_SHOT_VALUES_NON_TABULA
        ) + expand(
            "output_data/llm_{model}_{condition}_{shots}-shot_results.csv",
            model=LLM_MODEL_ID_TABULA, condition=CONDITIONS, shots=NUM_SHOT_VALUES_TABULA
        ),
        tabpfn_plots=expand(
            "output_plots/tabpfn_{condition}_{shots}-shot_roc_curve.png",
            condition=CONDITIONS, shots=NUM_SHOT_VALUES_TABPFN
        ),
        tabpfn_results=expand(
            "output_data/tabpfn_{condition}_{shots}-shot_results.csv",
            condition=CONDITIONS, shots=NUM_SHOT_VALUES_TABPFN
        ),
        results="output_data/evaluation_results.csv",
        summary_plot="output_plots/combined_publication_plots.png"

rule create_data:
    input:
        script="code/0_create_data.py",
        config="config.yaml"
    output:
        data=config["inputdata_path"] + "data.parquet"
    threads: 1
    run:
        shell(f"python {input.script} --config '{input.config}' --output '{output.data}'")

rule process_data:
    input:
        script="code/1_process_data.py",
        raw_data=config["inputdata_path"] + "data.parquet"
    output:
        processed_data="output_data/processed_data_{condition}.parquet"
    threads: 1
    run:
        args = []
        if wildcards.condition == "minmax":         args.append("--minmax")
        if wildcards.condition == "stats":          args.append("--stats")
        if wildcards.condition == "hourly":         args.append("--hourly")
        if wildcards.condition == "forward_fill":   args.extend(["--hourly", "--forward-fill"])
        if wildcards.condition == "hourly_stats":   args.extend(["--hourly", "--stats"])
        shell(f"python {input.script} --input '{input.raw_data}' --output '{output.processed_data}' {' '.join(args)}")

rule baseline:
    input:
        script=lambda wildcards: f"code/2_baseline_{wildcards.baseline}.py",
        data="output_data/processed_data_{condition}.parquet"
    output:
        plot="output_plots/baseline_{baseline}_{condition}_roc_curve.png",
        results="output_data/baseline_{baseline}_{condition}_results.csv"
    threads: 1
    priority: 20
    run:
        output_dir = os.path.dirname(output.results)
        plot_dir = os.path.dirname(output.plot)
        shell(f"python {input.script} --input '{input.data}' --output_dir '{output_dir}' --plot_dir '{plot_dir}'")
        shell("touch {output.plot} {output.results}")

rule baseline_oasis:
    input:
        script="code/2_baseline_oasis.py",
        data=config["inputdata_path"] + "data.parquet"
    output:
        plot="output_plots/baseline_oasis_basic_roc_curve.png",
        results="output_data/baseline_oasis_basic_results.csv"
    threads: 1
    priority: 20
    run:
        output_dir = os.path.dirname(output.results)
        plot_dir = os.path.dirname(output.plot)
        shell(f"python {input.script} --input '{input.data}' --output_dir '{output_dir}' --plot_dir '{plot_dir}'")
        shell("touch {output.plot} {output.results}")

rule baseline_tabpfn:
    input:
        script="code/3_tabpfn.py",
        data="output_data/processed_data_{condition}.parquet"
    output:
        plot="output_plots/baseline_tabpfn_{condition}_roc_curve.png",
        results="output_data/baseline_tabpfn_{condition}_results.csv"
    threads: 1
    priority: 20
    run:
        output_dir = os.path.dirname(output.results)
        plot_dir = os.path.dirname(output.plot)
        shell(f"python {input.script} --input '{input.data}' --output_dir '{output_dir}' --plot_dir '{plot_dir}'")
        shell("touch {output.plot} {output.results}")

rule serialize_data:
    input:
        script="code/4_serialize_data.py",
        data="output_data/processed_data_{condition}.parquet"
    output:
        serialized_train_set="output_data/serialized_data_{condition}.txt",
        serialized_test_set="output_data/serialized_data_{condition}_test.txt"
    params:
        condition=lambda wildcards: wildcards.condition
    threads: 1
    run:
        serialized_dir = os.path.dirname(output.serialized_train_set)
        shell(f"python {input.script} --input '{input.data}' --output_dir '{serialized_dir}' --condition {params.condition}")
        shell("touch {output.serialized_train_set} {output.serialized_test_set}")

rule evaluate_llm_with_shots:
    input:
        script="code/5_evaluate_LLM.py",
        serialized_data="output_data/serialized_data_{condition}.txt",
        serialized_data_test="output_data/serialized_data_{condition}_test.txt",
        processed_data="output_data/processed_data_{condition}.parquet"
    output:
        results="output_data/llm_{model}_{condition}_{shots}-shot_results.csv",
        plot="output_plots/llm_{model}_{condition}_{shots}-shot_roc_curve.png"
    threads: 1
    priority: 10
    run:
        output_dir = os.path.dirname(output.results)
        plot_dir = os.path.dirname(output.plot)
        shell(
            f"python {input.script} "
            f"--serialized_data_path '{input.serialized_data}' "
            f"--serialized_data_test_path '{input.serialized_data_test}' "
            f"--processed_data_path '{input.processed_data}' "
            f"--output_dir '{output_dir}' "
            f"--plot_dir '{plot_dir}' "
            f"--model '{wildcards.model}' "
            f"--condition '{wildcards.condition}' "
            f"--num_shots {wildcards.shots}"
        )
        shell("touch {output.plot} {output.results}")

rule evaluate_tabpfn_with_shots:
    input:
        script="code/5_evaluate_TabPFN.py",
        processed_data="output_data/processed_data_{condition}.parquet"
    output:
        results="output_data/tabpfn_{condition}_{shots}-shot_results.csv",
        plot="output_plots/tabpfn_{condition}_{shots}-shot_roc_curve.png"
    threads: 1
    priority: 100
    run:
        output_dir = os.path.dirname(output.results)
        plot_dir = os.path.dirname(output.plot)
        args = []
        if wildcards.condition == "minmax":         args.append("--minmax")
        if wildcards.condition == "stats":          args.append("--stats")
        if wildcards.condition == "hourly":         args.append("--hourly")
        if wildcards.condition == "forward_fill":   args.extend(["--hourly", "--forward-fill"])
        if wildcards.condition == "hourly_stats":   args.extend(["--hourly", "--stats"])
        shell(
            f"python {input.script} "
            f"--processed_data_path '{input.processed_data}' "
            f"--output_dir '{output_dir}' "
            f"--plot_dir '{plot_dir}' "
            f"--num_shots {wildcards.shots} "
            f"{' '.join(args)}"
        )
        shell("touch {output.plot} {output.results}")

rule aggregate_results:
    input:
        script="code/aggregate_results.py",
        baseline_results=expand(
            "output_data/baseline_{baseline}_{condition}_results.csv",
            baseline=BASELINE_MODELS, condition=CONDITIONS
        ) + expand(
            "output_data/baseline_tabpfn_{condition}_results.csv",
            condition=CONDITIONS
        ) + ["output_data/baseline_oasis_basic_results.csv"],
        llm_results=expand(
            "output_data/llm_{model}_{condition}_{shots}-shot_results.csv",
            model=LLM_MODEL_IDS_NON_TABULA, condition=CONDITIONS, shots=NUM_SHOT_VALUES_NON_TABULA
        ) + expand(
            "output_data/llm_{model}_{condition}_{shots}-shot_results.csv",
            model=LLM_MODEL_ID_TABULA, condition=CONDITIONS, shots=NUM_SHOT_VALUES_TABULA
        ),
        tabpfn_results=expand(
            "output_data/tabpfn_{condition}_{shots}-shot_results.csv",
            condition=CONDITIONS, shots=NUM_SHOT_VALUES_TABPFN
        ),
    output:
        csv="output_data/evaluation_results.csv"
    threads: 1
    run:
        input_dir = os.path.dirname(input.baseline_results[0])
        shell(
            f"python {input.script} "
            f"--input_dir '{input_dir}' "
            f"--output_csv '{output.csv}'"
        )
        shell("touch {output.csv}")

rule plot_combined_publication:
    input:
        script="code/6_combined_publication_plots.py",
        results_csv="output_data/evaluation_results.csv"
    output:
        plot="output_plots/combined_publication_plots.png"
    threads: 1
    run:
        output_dir = os.path.dirname(output.plot)
        shell(
            f"python {input.script} "
            f"--input_csv {input.results_csv} "
            f"--output_dir '{output_dir}'"
        )
        shell("touch {output.plot}")