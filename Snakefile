import os

configfile: "config.yaml"

# Define LLM models and number of shots
LLM_MODEL_IDS = ["qwen_3_8b"] # , "qwen_3_8b_think"] #, "medgemma_3_4b"]
# NUM_SHOT_VALUES will be [0, 1, 2, 4, 8, 16, 32, 64, 128, 256] (2^0 to 2^8)
NUM_SHOT_VALUES = [0] + [2**i for i in range(5)]
NUM_SHOT_VALUES_TABPFN = [2**i for i in range(9)]
NUM_SHOT_VALUES_TABPFN_TS = [2**i for i in range(5)]

rule all:
    input:
        processed_data="output_data/processed_data.parquet",
        processed_data_hourly="output_data/processed_data_hourly.parquet",
        processed_data_forward_fill="output_data/processed_data_forward_fill.parquet",
        processed_data_stats="output_data/processed_data_stats.parquet",
        processed_data_minmax="output_data/processed_data_minmax.parquet",
        processed_data_hourly_stats="output_data/processed_data_hourly_stats.parquet",
        processed_data_hourly_long="output_data/processed_data_hourly_long.parquet",
        processed_data_survival="output_data/processed_data_survival.parquet",
        log_summary="output_data/baseline_logistic_summary.txt",
        log_plot="output_plots/baseline_logistic_roc_curve.png",
        log_hourly_summary="output_data/baseline_logistic_hourly_summary.txt",
        log_hourly_plot="output_plots/baseline_logistic_hourly_roc_curve.png",
        log_forward_fill_summary="output_data/baseline_logistic_forward_fill_summary.txt",
        log_forward_fill_plot="output_plots/baseline_logistic_forward_fill_roc_curve.png",
        log_stats_summary="output_data/baseline_logistic_stats_summary.txt",
        log_stats_plot="output_plots/baseline_logistic_stats_roc_curve.png",
        log_minmax_summary="output_data/baseline_logistic_minmax_summary.txt",
        log_minmax_plot="output_plots/baseline_logistic_minmax_roc_curve.png",
        log_hourly_stats_summary="output_data/baseline_logistic_hourly_stats_summary.txt",
        log_hourly_stats_plot="output_plots/baseline_logistic_hourly_stats_roc_curve.png",
        xgboost_plot="output_plots/baseline_xgboost_roc_curve.png",
        xgboost_hourly_plot="output_plots/baseline_xgboost_hourly_roc_curve.png",
        xgboost_forward_fill_plot="output_plots/baseline_xgboost_forward_fill_roc_curve.png",
        xgboost_stats_plot="output_plots/baseline_xgboost_stats_roc_curve.png",
        xgboost_minmax_plot="output_plots/baseline_xgboost_minmax_roc_curve.png",
        xgboost_hourly_stats_plot="output_plots/baseline_xgboost_hourly_stats_roc_curve.png",
        oasis_plot="output_plots/baseline_oasis_roc_curve.png",
        tabpfn_plot="output_plots/tabpfn_roc_curve.png",
        serialized_data_train="output_data/serialized_data.txt",
        serialized_data_test="output_data/serialized_data_test.txt",
        results="output_data/evaluation_results.csv",
        llm_plots=expand(
            "output_plots/llm_{model}_{shots}-shot_roc_curve.png",
            model=LLM_MODEL_IDS,
            shots=NUM_SHOT_VALUES
        ),
        tabpfn_plots=expand(
            "output_plots/tabpfn_{shots}-shot_roc_curve.png",
            shots=NUM_SHOT_VALUES_TABPFN
        ),
        tabpfn_hourly_plots=expand(
            "output_plots/tabpfn-hourly_{shots}-shot_roc_curve.png",
            shots=NUM_SHOT_VALUES_TABPFN
        ),
        tabpfn_stats_plots=expand(
            "output_plots/tabpfn-stats_{shots}-shot_roc_curve.png",
            shots=NUM_SHOT_VALUES_TABPFN
        ),
        tabpfn_hourly_stats_plots=expand(
            "output_plots/tabpfn-hourly-stats_{shots}-shot_roc_curve.png",
            shots=NUM_SHOT_VALUES_TABPFN
        ),
        tabpfn_timeseries_plots=expand(
            "output_plots/tabpfn-timeseries_{shots}-shot_roc_curve.png",
            shots=NUM_SHOT_VALUES_TABPFN_TS
        ),
        tabpfn_survival_plots=expand(
            "output_plots/tabpfn-survival_{shots}-shot_roc_curve.png",
            shots=NUM_SHOT_VALUES_TABPFN_TS
        ),
        summary_plot_auc_model_shots="output_plots/roc_curves_faceted_by_shots.png",
        summary_plot_auc_shots_model="output_plots/roc_curves_faceted_by_model.png",
        summary_plot_metrics_evolution="output_plots/metrics_evolution_with_shots.png",

rule create_data:
    input:
        script="code/0_create_data.py",
        config="config.yaml"
        # Note: Input parquet files are implicitly used by the script
        # Add them explicitly if needed for stricter dependency tracking
        # info=config["reprodicu_path"] + "patient_information.parquet",
        # vitals=config["reprodicu_path"] + "timeseries_vitals.parquet",
        # labs=config["reprodicu_path"] + "timeseries_labs.parquet",
        # resp=config["reprodicu_path"] + "timeseries_resp.parquet",
        # inout=config["reprodicu_path"] + "timeseries_intakeoutput.parquet"
    output:
        data=config["inputdata_path"] + "data.parquet"
    threads: 1
    run:
        # Use shell() function with f-string
        shell(f"python {input.script} --config '{input.config}' --output '{output.data}'")

rule process_data:
    input:
        script="code/1_process_data.py",
        raw_data=config["inputdata_path"] + "data.parquet"
    output:
        processed_data="output_data/processed_data.parquet"
    threads: 1
    run:
        # Use shell() function with f-string
        shell(f"python {input.script} --input '{input.raw_data}' --output '{output.processed_data}'")

rule process_data_hourly:
    input:
        script="code/1_process_data.py",
        raw_data=config["inputdata_path"] + "data.parquet"
    output:
        processed_data_hourly="output_data/processed_data_hourly.parquet"
    threads: 1
    run:
        # Use shell() function with f-string
        shell(f"python {input.script} --input '{input.raw_data}' --output '{output.processed_data_hourly}' --hourly")

rule process_data_forward_fill:
    input:
        script="code/1_process_data.py",
        raw_data=config["inputdata_path"] + "data.parquet"
    output:
        processed_data_forward_fill="output_data/processed_data_forward_fill.parquet"
    threads: 1
    run:
        shell(f"python {input.script} --input '{input.raw_data}' --output '{output.processed_data_forward_fill}' --hourly --forward-fill")

rule process_data_stats:
    input:
        script="code/1_process_data.py",
        raw_data=config["inputdata_path"] + "data.parquet"
    output:
        processed_data_stats="output_data/processed_data_stats.parquet"
    threads: 1
    run:
        # Use shell() function with f-string
        shell(f"python {input.script} --input '{input.raw_data}' --output '{output.processed_data_stats}' --stats")

rule process_data_minmax:
    input:
        script="code/1_process_data.py",
        raw_data=config["inputdata_path"] + "data.parquet"
    output:
        processed_data_minmax="output_data/processed_data_minmax.parquet"
    threads: 1
    run:
        shell(f"python {input.script} --input '{input.raw_data}' --output '{output.processed_data_minmax}' --minmax")

rule process_data_hourly_stats:
    input:
        script="code/1_process_data.py",
        raw_data=config["inputdata_path"] + "data.parquet"
    output:
        processed_data_hourly_stats="output_data/processed_data_hourly_stats.parquet"
    threads: 1
    run:
        # Use shell() function with f-string
        shell(f"python {input.script} --input '{input.raw_data}' --output '{output.processed_data_hourly_stats}' --hourly --stats")

rule process_data_hourly_long:
    input:
        script="code/1_process_data.py",
        raw_data=config["inputdata_path"] + "data.parquet"
    output:
        processed_data_hourly_long="output_data/processed_data_hourly_long.parquet"
    threads: 1
    run:
        # Use shell() function with f-string
        shell(f"python {input.script} --input '{input.raw_data}' --output '{output.processed_data_hourly_long}' --hourly-long")

rule process_data_survival:
    input:
        script="code/1_process_data.py",
        raw_data=config["inputdata_path"] + "data.parquet"
    output:
        processed_data_survival="output_data/processed_data_survival.parquet"
    threads: 1
    run:
        # Use shell() function with f-string
        shell(f"python {input.script} --input '{input.raw_data}' --output '{output.processed_data_survival}' --survival")

rule baseline_logistic:
    input:
        script="code/2_baseline_logistic.py",
        data="output_data/processed_data.parquet"
    output:
        summary="output_data/baseline_logistic_summary.txt",
        plot="output_plots/baseline_logistic_roc_curve.png"
    threads: 1
    run:
        # Calculate directories within the run block
        output_dir = os.path.dirname(output.summary)
        plot_dir = os.path.dirname(output.plot)
        # Use shell() function with f-string
        shell(f"python {input.script} --input '{input.data}' --output_dir '{output_dir}' --plot_dir '{plot_dir}'")

rule baseline_logistic_hourly:
    input:
        script="code/2_baseline_logistic.py",
        data="output_data/processed_data_hourly.parquet"
    output:
        summary="output_data/baseline_logistic_hourly_summary.txt",
        plot="output_plots/baseline_logistic_hourly_roc_curve.png"
    threads: 1
    run:
        output_dir = os.path.dirname(output.summary)
        plot_dir = os.path.dirname(output.plot)
        shell(f"python {input.script} --input '{input.data}' --output_dir '{output_dir}' --plot_dir '{plot_dir}'")

rule baseline_logistic_forward_fill:
    input:
        script="code/2_baseline_logistic.py",
        data="output_data/processed_data_forward_fill.parquet"
    output:
        summary="output_data/baseline_logistic_forward_fill_summary.txt",
        plot="output_plots/baseline_logistic_forward_fill_roc_curve.png"
    threads: 1
    run:
        output_dir = os.path.dirname(output.summary)
        plot_dir = os.path.dirname(output.plot)
        shell(f"python {input.script} --input '{input.data}' --output_dir '{output_dir}' --plot_dir '{plot_dir}'")

rule baseline_logistic_stats:
    input:
        script="code/2_baseline_logistic.py",
        data="output_data/processed_data_stats.parquet"
    output:
        summary="output_data/baseline_logistic_stats_summary.txt",
        plot="output_plots/baseline_logistic_stats_roc_curve.png"
    threads: 1
    run:
        output_dir = os.path.dirname(output.summary)
        plot_dir = os.path.dirname(output.plot)
        shell(f"python {input.script} --input '{input.data}' --output_dir '{output_dir}' --plot_dir '{plot_dir}'")

rule baseline_logistic_minmax:
    input:
        script="code/2_baseline_logistic.py",
        data="output_data/processed_data_minmax.parquet"
    output:
        summary="output_data/baseline_logistic_minmax_summary.txt",
        plot="output_plots/baseline_logistic_minmax_roc_curve.png"
    threads: 1
    run:
        output_dir = os.path.dirname(output.summary)
        plot_dir = os.path.dirname(output.plot)
        shell(f"python {input.script} --input '{input.data}' --output_dir '{output_dir}' --plot_dir '{plot_dir}'")

rule baseline_logistic_hourly_stats:
    input:
        script="code/2_baseline_logistic.py",
        data="output_data/processed_data_hourly_stats.parquet"
    output:
        summary="output_data/baseline_logistic_hourly_stats_summary.txt",
        plot="output_plots/baseline_logistic_hourly_stats_roc_curve.png"
    threads: 1
    run:
        output_dir = os.path.dirname(output.summary)
        plot_dir = os.path.dirname(output.plot)
        shell(f"python {input.script} --input '{input.data}' --output_dir '{output_dir}' --plot_dir '{plot_dir}'")

rule baseline_xgboost:
    input:
        script="code/2_baseline_xgboost.py",
        data="output_data/processed_data.parquet"
    output:
        model="output_data/baseline_xgboost_model.json",
        plot="output_plots/baseline_xgboost_roc_curve.png"
    threads: 1
    run:
        # Calculate directories within the run block
        output_dir = os.path.dirname(output.model)
        plot_dir = os.path.dirname(output.plot)
        # Use shell() function with f-string
        shell(f"python {input.script} --input '{input.data}' --output_dir '{output_dir}' --plot_dir '{plot_dir}'")

rule baseline_xgboost_hourly:
    input:
        script="code/2_baseline_xgboost.py",
        data="output_data/processed_data_hourly.parquet"
    output:
        model="output_data/baseline_xgboost_hourly_model.json",
        plot="output_plots/baseline_xgboost_hourly_roc_curve.png"
    threads: 1
    run:
        output_dir = os.path.dirname(output.model)
        plot_dir = os.path.dirname(output.plot)
        shell(f"python {input.script} --input '{input.data}' --output_dir '{output_dir}' --plot_dir '{plot_dir}'")

rule baseline_xgboost_forward_fill:
    input:
        script="code/2_baseline_xgboost.py",
        data="output_data/processed_data_forward_fill.parquet"
    output:
        model="output_data/baseline_xgboost_forward_fill_model.json",
        plot="output_plots/baseline_xgboost_forward_fill_roc_curve.png"
    threads: 1
    run:
        output_dir = os.path.dirname(output.model)
        plot_dir = os.path.dirname(output.plot)
        shell(f"python {input.script} --input '{input.data}' --output_dir '{output_dir}' --plot_dir '{plot_dir}'")

rule baseline_xgboost_stats:
    input:
        script="code/2_baseline_xgboost.py",
        data="output_data/processed_data_stats.parquet"
    output:
        model="output_data/baseline_xgboost_stats_model.json",
        plot="output_plots/baseline_xgboost_stats_roc_curve.png"
    threads: 1
    run:
        output_dir = os.path.dirname(output.model)
        plot_dir = os.path.dirname(output.plot)
        shell(f"python {input.script} --input '{input.data}' --output_dir '{output_dir}' --plot_dir '{plot_dir}'")

rule baseline_xgboost_minmax:
    input:
        script="code/2_baseline_xgboost.py",
        data="output_data/processed_data_minmax.parquet"
    output:
        model="output_data/baseline_xgboost_minmax_model.json",
        plot="output_plots/baseline_xgboost_minmax_roc_curve.png"
    threads: 1
    run:
        output_dir = os.path.dirname(output.model)
        plot_dir = os.path.dirname(output.plot)
        shell(f"python {input.script} --input '{input.data}' --output_dir '{output_dir}' --plot_dir '{plot_dir}'")

rule baseline_xgboost_hourly_stats:
    input:
        script="code/2_baseline_xgboost.py",
        data="output_data/processed_data_hourly_stats.parquet"
    output:
        model="output_data/baseline_xgboost_hourly_stats_model.json",
        plot="output_plots/baseline_xgboost_hourly_stats_roc_curve.png"
    threads: 1
    run:
        output_dir = os.path.dirname(output.model)
        plot_dir = os.path.dirname(output.plot)
        shell(f"python {input.script} --input '{input.data}' --output_dir '{output_dir}' --plot_dir '{plot_dir}'")

rule baseline_oasis:
    input:
        script="code/2_baseline_oasis.py",
        data=config["inputdata_path"] + "data.parquet"
    output:
        plot="output_plots/baseline_oasis_roc_curve.png"
    threads: 1
    run:
        plot_dir = os.path.dirname(output.plot)
        shell(f"python {input.script} --input '{input.data}' --plot_dir '{plot_dir}'")

rule baseline_tabpfn:
    input:
        script="code/3_tabpfn.py",
        data="output_data/processed_data.parquet"
    output:
        plot="output_plots/tabpfn_roc_curve.png"
    threads: 1
    run:
        plot_dir = os.path.dirname(output.plot)
        shell(f"python {input.script} --input '{input.data}' --plot_dir '{plot_dir}'")

rule baseline_tabpfn_hourly:
    input:
        script="code/3_tabpfn.py",
        data="output_data/processed_data_hourly.parquet"
    output:
        plot="output_plots/tabpfn_hourly_roc_curve.png"
    threads: 1
    run:
        plot_dir = os.path.dirname(output.plot)
        shell(f"python {input.script} --input '{input.data}' --plot_dir '{plot_dir}'")

rule baseline_tabpfn_forward_fill:
    input:
        script="code/3_tabpfn.py",
        data="output_data/processed_data_forward_fill.parquet"
    output:
        plot="output_plots/tabpfn_forward_fill_roc_curve.png"
    threads: 1
    run:
        plot_dir = os.path.dirname(output.plot)
        shell(f"python {input.script} --input '{input.data}' --plot_dir '{plot_dir}'")

rule baseline_tabpfn_stats:
    input:
        script="code/3_tabpfn.py",
        data="output_data/processed_data_stats.parquet"
    output:
        plot="output_plots/tabpfn_stats_roc_curve.png"
    threads: 1
    run:
        plot_dir = os.path.dirname(output.plot)
        shell(f"python {input.script} --input '{input.data}' --plot_dir '{plot_dir}'")

rule baseline_tabpfn_minmax:
    input:
        script="code/3_tabpfn.py",
        data="output_data/processed_data_minmax.parquet"
    output:
        plot="output_plots/tabpfn_minmax_roc_curve.png"
    threads: 1
    run:
        plot_dir = os.path.dirname(output.plot)
        shell(f"python {input.script} --input '{input.data}' --plot_dir '{plot_dir}'")

rule serialize_data:
    input:
        script="code/4_serialize_data.py",
        data="output_data/processed_data.parquet"
    output:
        serialized_train_set="output_data/serialized_data.txt",
        serialized_test_set="output_data/serialized_data_test.txt",
        serialized_tabula_train="output_data/tabula_serialized_data.txt",
        serialized_tabula_test="output_data/tabula_serialized_data_test.txt"
    threads: 1
    run:
        # Calculate directory within the run block
        serialized_dir = os.path.dirname(output.serialized_train_set)
        # Use shell() function with f-string
        shell(f"python {input.script} --input '{input.data}' --output_dir '{serialized_dir}'")

rule evaluate_llm_with_shots:
    input:
        script="code/5_evaluate_LLM.py",
        serialized_data=lambda wildcards: "output_data/tabula_serialized_data.txt" if wildcards.model == "tabula_8b" else "output_data/serialized_data.txt",
        serialized_data_test=lambda wildcards: "output_data/tabula_serialized_data_test.txt" if wildcards.model == "tabula_8b" else "output_data/serialized_data_test.txt",
        processed_data="output_data/processed_data.parquet"
    output:
        results="output_data/llm_{model}_{shots}-shot_results.csv",
        plot="output_plots/llm_{model}_{shots}-shot_roc_curve.png"
    threads: 1
    run:
        model_wc = wildcards.model
        shots_wc = wildcards.shots

        # Calculate directories within the run block
        output_dir = os.path.dirname(output.results)
        plot_dir = os.path.dirname(output.plot)

        # Run the script, directing its output to the temporary directories
        shell(
            f"python {input.script} "
            f"--serialized_data_path '{input.serialized_data}' "
            f"--serialized_data_test_path '{input.serialized_data_test}' "
            f"--processed_data_path '{input.processed_data}' "
            f"--output_dir '{output_dir}' "
            f"--plot_dir '{plot_dir}' "
            f"--model '{model_wc}' "
            f"--num_shots {shots_wc}"
        )

rule evaluate_tabpfn_with_shots:
    input:
        script="code/5_evaluate_TabPFN.py",
        processed_data="output_data/processed_data.parquet"
    output:
        results="output_data/tabpfn_{shots}-shot_results.csv",
        plot="output_plots/tabpfn_{shots}-shot_roc_curve.png"
    threads: 1
    run:
        shots_wc = wildcards.shots
        output_dir = os.path.dirname(output.results)
        plot_dir = os.path.dirname(output.plot)

        shell(
            f"python {input.script} "
            f"--processed_data_path '{input.processed_data}' "
            f"--output_dir '{output_dir}' "
            f"--plot_dir '{plot_dir}' "
            f"--num_shots {shots_wc}"
        )

rule evaluate_tabpfn_hourly_with_shots:
    input:
        script="code/5_evaluate_TabPFN.py",
        processed_data="output_data/processed_data_hourly.parquet"
    output:
        results="output_data/tabpfn-hourly_{shots}-shot_results.csv",
        plot="output_plots/tabpfn-hourly_{shots}-shot_roc_curve.png"
    threads: 1
    run:
        shots_wc = wildcards.shots
        output_dir = os.path.dirname(output.results)
        plot_dir = os.path.dirname(output.plot)

        shell(
            f"python {input.script} "
            f"--processed_data_path '{input.processed_data}' "
            f"--output_dir '{output_dir}' "
            f"--plot_dir '{plot_dir}' "
            f"--num_shots {shots_wc} "
            f"--hourly"
        )

rule evaluate_tabpfn_stats_with_shots:
    input:
        script="code/5_evaluate_TabPFN.py",
        processed_data="output_data/processed_data_stats.parquet"
    output:
        results="output_data/tabpfn-stats_{shots}-shot_results.csv",
        plot="output_plots/tabpfn-stats_{shots}-shot_roc_curve.png"
    threads: 1
    run:
        shots_wc = wildcards.shots
        output_dir = os.path.dirname(output.results)
        plot_dir = os.path.dirname(output.plot)

        shell(
            f"python {input.script} "
            f"--processed_data_path '{input.processed_data}' "
            f"--output_dir '{output_dir}' "
            f"--plot_dir '{plot_dir}' "
            f"--num_shots {shots_wc} "
            f"--stats"
        )

rule evaluate_tabpfn_hourly_stats_with_shots:
    input:
        script="code/5_evaluate_TabPFN.py",
        processed_data="output_data/processed_data_hourly_stats.parquet"
    output:
        results="output_data/tabpfn-hourly-stats_{shots}-shot_results.csv",
        plot="output_plots/tabpfn-hourly-stats_{shots}-shot_roc_curve.png"
    threads: 1
    run:
        shots_wc = wildcards.shots
        output_dir = os.path.dirname(output.results)
        plot_dir = os.path.dirname(output.plot)

        shell(
            f"python {input.script} "
            f"--processed_data_path '{input.processed_data}' "
            f"--output_dir '{output_dir}' "
            f"--plot_dir '{plot_dir}' "
            f"--num_shots {shots_wc} "
            f"--hourly --stats"
        )

rule evaluate_tabpfn_timeseries_with_shots:
    input:
        script="code/5_evaluate_TabPFN_TimeSeries.py",
        processed_data="output_data/processed_data_hourly_long.parquet"
    output:
        results="output_data/tabpfn-timeseries_{shots}-shot_results.csv",
        plot="output_plots/tabpfn-timeseries_{shots}-shot_roc_curve.png"
    threads: 1
    run:
        shots_wc = wildcards.shots
        output_dir = os.path.dirname(output.results)
        plot_dir = os.path.dirname(output.plot)

        shell(
            f"python {input.script} "
            f"--processed_data_path '{input.processed_data}' "
            f"--output_dir '{output_dir}' "
            f"--plot_dir '{plot_dir}' "
            f"--num_shots {shots_wc} "
        )

rule evaluate_tabpfn_survival_with_shots:
    input:
        script="code/5_evaluate_TabPFN_TimeSeries.py",
        processed_data="output_data/processed_data_survival.parquet"
    output:
        results="output_data/tabpfn-survival_{shots}-shot_results.csv",
        plot="output_plots/tabpfn-survival_{shots}-shot_roc_curve.png"
    threads: 1
    run:
        shots_wc = wildcards.shots
        output_dir = os.path.dirname(output.results)
        plot_dir = os.path.dirname(output.plot)

        shell(
            f"python {input.script} "
            f"--processed_data_path '{input.processed_data}' "
            f"--output_dir '{output_dir}' "
            f"--plot_dir '{plot_dir}' "
            f"--num_shots {shots_wc} "
            f"--survival"
        )

rule evaluate_tabpfn_minmax_with_shots:
    input:
        script="code/5_evaluate_TabPFN.py",
        processed_data="output_data/processed_data_minmax.parquet"
    output:
        results="output_data/tabpfn-minmax_{shots}-shot_results.csv",
        plot="output_plots/tabpfn-minmax_{shots}-shot_roc_curve.png"
    threads: 1
    run:
        shots_wc = wildcards.shots
        output_dir = os.path.dirname(output.results)
        plot_dir = os.path.dirname(output.plot)

        shell(
            f"python {input.script} "
            f"--processed_data_path '{input.processed_data}' "
            f"--output_dir '{output_dir}' "
            f"--plot_dir '{plot_dir}' "
            f"--num_shots {shots_wc} "
            f"--minmax"
        )

rule aggregate_results:
    input:
        llm_results=expand(
            "output_data/llm_{model}_{shots}-shot_results.csv",
            model=LLM_MODEL_IDS,
            shots=NUM_SHOT_VALUES
        ),
        tabpfn_results=expand(
            "output_data/tabpfn_{shots}-shot_results.csv",
            shots=NUM_SHOT_VALUES_TABPFN
        ),
        tabpfn_hourly_results=expand(
            "output_data/tabpfn-hourly_{shots}-shot_results.csv",
            shots=NUM_SHOT_VALUES_TABPFN
        ),
        tabpfn_forward_fill_results=expand(
            "output_data/tabpfn-forward-fill_{shots}-shot_results.csv",
            shots=NUM_SHOT_VALUES_TABPFN
        ),
        tabpfn_stats_results=expand(
            "output_data/tabpfn-stats_{shots}-shot_results.csv",
            shots=NUM_SHOT_VALUES_TABPFN
        ),
        tabpfn_minmax_results=expand(
            "output_data/tabpfn-minmax_{shots}-shot_results.csv",
            shots=NUM_SHOT_VALUES_TABPFN
        ),
        tabpfn_hourly_stats_results=expand(
            "output_data/tabpfn-hourly-stats_{shots}-shot_results.csv",
            shots=NUM_SHOT_VALUES_TABPFN
        ),
        tabpfn_timeseries_results=expand(
            "output_data/tabpfn-timeseries_{shots}-shot_results.csv",
            shots=NUM_SHOT_VALUES_TABPFN_TS
        ),
        tabpfn_survival_results=expand(
            "output_data/tabpfn-survival_{shots}-shot_results.csv",
            shots=NUM_SHOT_VALUES_TABPFN_TS
        )
    output:
        aggregated_results="output_data/evaluation_results.csv"
    threads: 1
    run:
        import pandas as pd
        import os

        # Combine all input file paths into a single list
        all_input_files = (list(input.llm_results) + list(input.tabpfn_results) + 
                          list(input.tabpfn_hourly_results) + list(input.tabpfn_forward_fill_results) + 
                          list(input.tabpfn_stats_results) + list(input.tabpfn_minmax_results) + 
                          list(input.tabpfn_hourly_stats_results) + list(input.tabpfn_timeseries_results) + 
                          list(input.tabpfn_survival_results))
        
        dataframes_to_concat = []

        for file_path in all_input_files:
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                df = pd.read_csv(file_path)
                dataframes_to_concat.append(df)

        # Aggregate the DataFrames into a single DataFrame
        # Write the aggregated DataFrame to the output CSV file without the DataFrame index.
        aggregated_df = pd.concat(dataframes_to_concat, ignore_index=True)
        aggregated_df.to_csv(output.aggregated_results, index=False)
        print(f"Successfully aggregated {len(dataframes_to_concat)} CSV files into {output.aggregated_results}.")

rule tabpfn_all:
    input:
        processed_data="output_data/processed_data.parquet",
        processed_data_hourly="output_data/processed_data_hourly.parquet", 
        processed_data_forward_fill="output_data/processed_data_forward_fill.parquet",
        processed_data_stats="output_data/processed_data_stats.parquet",
        processed_data_minmax="output_data/processed_data_minmax.parquet",
        processed_data_hourly_stats="output_data/processed_data_hourly_stats.parquet",
        processed_data_hourly_long="output_data/processed_data_hourly_long.parquet",
        tabpfn_plots=expand(
            "output_plots/tabpfn_{shots}-shot_roc_curve.png",
            shots=NUM_SHOT_VALUES_TABPFN
        ),
        tabpfn_hourly_plots=expand(
            "output_plots/tabpfn-hourly_{shots}-shot_roc_curve.png",
            shots=NUM_SHOT_VALUES_TABPFN
        ),
        tabpfn_forward_fill_plots=expand(
            "output_plots/tabpfn-forward-fill_{shots}-shot_roc_curve.png",
            shots=NUM_SHOT_VALUES_TABPFN
        ),
        tabpfn_stats_plots=expand(
            "output_plots/tabpfn-stats_{shots}-shot_roc_curve.png",
            shots=NUM_SHOT_VALUES_TABPFN
        ),
        tabpfn_minmax_plots=expand(
            "output_plots/tabpfn-minmax_{shots}-shot_roc_curve.png",
            shots=NUM_SHOT_VALUES_TABPFN
        ),
        tabpfn_hourly_stats_plots=expand(
            "output_plots/tabpfn-hourly-stats_{shots}-shot_roc_curve.png",
            shots=NUM_SHOT_VALUES_TABPFN
        ),
        tabpfn_timeseries_plots=expand(
            "output_plots/tabpfn-timeseries_{shots}-shot_roc_curve.png",
            shots=NUM_SHOT_VALUES_TABPFN_TS
        ),
        tabpfn_survival_plots=expand(
            "output_plots/tabpfn-survival_{shots}-shot_roc_curve.png",
            shots=NUM_SHOT_VALUES_TABPFN_TS
        )

rule baselines_all:
    input:
        # Data processing
        processed_data="output_data/processed_data.parquet",
        processed_data_hourly="output_data/processed_data_hourly.parquet",
        processed_data_forward_fill="output_data/processed_data_forward_fill.parquet",
        processed_data_stats="output_data/processed_data_stats.parquet",
        processed_data_minmax="output_data/processed_data_minmax.parquet",
        processed_data_hourly_stats="output_data/processed_data_hourly_stats.parquet",
        # Logistic regression baselines
        log_summary="output_data/baseline_logistic_summary.txt",
        log_plot="output_plots/baseline_logistic_roc_curve.png",
        log_hourly_summary="output_data/baseline_logistic_hourly_summary.txt",
        log_hourly_plot="output_plots/baseline_logistic_hourly_roc_curve.png",
        log_forward_fill_summary="output_data/baseline_logistic_forward_fill_summary.txt",
        log_forward_fill_plot="output_plots/baseline_logistic_forward_fill_roc_curve.png",
        log_stats_summary="output_data/baseline_logistic_stats_summary.txt",
        log_stats_plot="output_plots/baseline_logistic_stats_roc_curve.png",
        log_minmax_summary="output_data/baseline_logistic_minmax_summary.txt",
        log_minmax_plot="output_plots/baseline_logistic_minmax_roc_curve.png",
        log_hourly_stats_summary="output_data/baseline_logistic_hourly_stats_summary.txt",
        log_hourly_stats_plot="output_plots/baseline_logistic_hourly_stats_roc_curve.png",
        # XGBoost baselines
        xgboost_plot="output_plots/baseline_xgboost_roc_curve.png",
        xgboost_hourly_plot="output_plots/baseline_xgboost_hourly_roc_curve.png",
        xgboost_forward_fill_plot="output_plots/baseline_xgboost_forward_fill_roc_curve.png",
        xgboost_stats_plot="output_plots/baseline_xgboost_stats_roc_curve.png",
        xgboost_minmax_plot="output_plots/baseline_xgboost_minmax_roc_curve.png",
        xgboost_hourly_stats_plot="output_plots/baseline_xgboost_hourly_stats_roc_curve.png",
        # TabPFN baselines
        tabpfn_plot="output_plots/tabpfn_roc_curve.png",
        tabpfn_hourly_plot="output_plots/tabpfn_hourly_roc_curve.png",
        tabpfn_forward_fill_plot="output_plots/tabpfn_forward_fill_roc_curve.png",
        tabpfn_stats_plot="output_plots/tabpfn_stats_roc_curve.png",
        tabpfn_minmax_plot="output_plots/tabpfn_minmax_roc_curve.png",
        tabpfn_hourly_stats_plot="output_plots/tabpfn_hourly_stats_roc_curve.png",
        # OASIS baseline
        oasis_plot="output_plots/baseline_oasis_roc_curve.png"