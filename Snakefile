import os

configfile: "config.yaml"

# Define LLM models and number of shots
LLM_MODEL_IDS = ["qwen_0_5b", "qwen_1_5b", "llama_3_2_1b"]
# NUM_SHOT_VALUES will be [1, 2, 4, 8, 16, 32, 64, 128, 256] (2^0 to 2^8)
NUM_SHOT_VALUES = [2**i for i in range(5)]

rule all:
    input:
        processed_data="output_data/processed_data.parquet",
        log_summary="output_data/baseline_logistic_summary.txt",
        log_plot="output_plots/baseline_logistic_roc_curve.png",
        xgboost_plot="output_plots/baseline_xgboost_roc_curve.png",
        oasis_plot="output_plots/baseline_oasis_roc_curve.png",
        tabpfn_plot="output_plots/tabpfn_roc_curve.png",
        serialized_data_train="output_data/serialized_data.txt",
        serialized_data_test="output_data/serialized_data_test.txt",
        llm_results="output_data/llm_evaluation_results.csv", # Aggregated results
        llm_plots=expand(
            "output_plots/llm_{model}_{shots}-shot_roc_curve.png",
            model=LLM_MODEL_IDS,
            shots=NUM_SHOT_VALUES
        ),

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

rule baseline_oasis:
    input:
        script="code/2_baseline_oasis.py",
        data=config["inputdata_path"] + "data.parquet"
    output:
        plot="output_plots/baseline_oasis_roc_curve.png"
    threads: 1
    run:
        # Calculate directory within the run block
        plot_dir = os.path.dirname(output.plot)
        # Use shell() function with f-string
        shell(f"python {input.script} --input '{input.data}' --plot_dir '{plot_dir}'")

rule tabpfn:
    input:
        script="code/3_tabpfn.py",
        data="output_data/processed_data.parquet"
    output:
        plot="output_plots/tabpfn_roc_curve.png"
    threads: 1
    run:
        # Calculate directories within the run block
        plot_dir = os.path.dirname(output.plot)
        # Use shell() function with f-string
        shell(f"python {input.script} --input '{input.data}' --plot_dir '{plot_dir}'")

rule serialize_data:
    input:
        script="code/4_serialize_data.py",
        data="output_data/processed_data.parquet"
    output:
        serialized_train_set="output_data/serialized_data.txt",
        serialized_test_set="output_data/serialized_data_test.txt"
    threads: 1
    run:
        # Calculate directory within the run block
        serialized_dir = os.path.dirname(output.serialized_train_set)
        # Use shell() function with f-string
        shell(f"python {input.script} --input '{input.data}' --output_dir '{serialized_dir}'")

rule evaluate_llm_with_shots:
    input:
        script="code/5_evaluate_LLM.py",
        serialized_data="output_data/serialized_data.txt",
        serialized_data_test="output_data/serialized_data_test.txt",
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

rule aggregate_llm_results:
    input:
        # Collect all individual results CSVs
        results_list=expand(
            "output_data/llm_{model}_{shots}-shot_results.csv",
            model=LLM_MODEL_IDS,
            shots=NUM_SHOT_VALUES
        )
    output:
        aggregated_results="output_data/llm_evaluation_results.csv"
    threads: 1
    run:
        # Simple aggregation: concatenate the CSVs.
        # A more sophisticated script could be used here for complex aggregation.
        with open(output.aggregated_results, 'w') as outfile:
            first_file = True
            # Iterate over the list of all generated result files
            for fname in input.results_list:
                # Check if file exists and is not empty, as some runs might fail
                if os.path.exists(fname) and os.path.getsize(fname) > 0:
                    with open(fname, 'r') as infile:
                        if first_file:
                            outfile.write(infile.read())
                            first_file = False
                        else:
                            try:
                                next(infile) # Skip header for subsequent files
                                outfile.write(infile.read())
                            except StopIteration: # Handles empty files after header
                                pass 
                else:
                    print(f"Warning: File {fname} not found or is empty, skipping aggregation.")