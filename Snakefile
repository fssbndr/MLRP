import os

configfile: "config.yaml"

rule all:
    input:
        processed_data="output_data/processed_data.parquet",
        log_summary="output_data/baseline_logistic_summary.txt",
        log_plot="output_plots/baseline_logistic_roc_curve.png",
        xgboost_plot="output_plots/baseline_xgboost_roc_curve.png",
        oasis_plot="output_plots/baseline_oasis_roc_curve.png",
        tabpfn_plot="output_plots/tabpfn_roc_curve.png",
        serialized_data="output_data/serialized_data.txt",
        llm_results="output_data/llm_evaluation_results.csv",
        llm_qwen0_5_plot="output_plots/llm_qwen_0_5b_roc_curve.png",
        llm_qwen1_5_plot="output_plots/llm_qwen_1_5b_roc_curve.png",
        llm_llama_1_plot="output_plots/llm_llama_3_2_1b_roc_curve.png",

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
        serialized_text="output_data/serialized_data.txt"
    threads: 1
    run:
        # Calculate directory within the run block
        serialized_dir = os.path.dirname(output.serialized_text)
        # Use shell() function with f-string
        shell(f"python {input.script} --input '{input.data}' --output_dir '{serialized_dir}'")

rule evaluate_llm_qwen_0_5b:
    input:
        script="code/5_evaluate_LLM.py",
        serialized_data="output_data/serialized_data.txt",
        processed_data="output_data/processed_data.parquet"
    output:
        results="output_data/llm_qwen_0_5b_results.csv",
        plot="output_plots/llm_qwen_0_5b_roc_curve.png"
    threads: 1
    run:
        model="qwen_0_5b"
        # Calculate directories within the run block
        output_dir = os.path.dirname(output.results)
        plot_dir = os.path.dirname(output.plot)
        # Use shell() function with f-string
        shell(f"python {input.script} --serialized_data_path '{input.serialized_data}' --processed_data_path '{input.processed_data}' --output_dir '{output_dir}' --plot_dir '{plot_dir}' --model {model}")

rule evaluate_llm_qwen_1_5b:
    input:
        script="code/5_evaluate_LLM.py",
        serialized_data="output_data/serialized_data.txt",
        processed_data="output_data/processed_data.parquet"
    output:
        results="output_data/llm_qwen_1_5b_results.csv",
        plot="output_plots/llm_qwen_1_5b_roc_curve.png"
    threads: 1
    run:
        model = "qwen_1_5b"
        # Calculate directories within the run block
        output_dir = os.path.dirname(output.results)
        plot_dir = os.path.dirname(output.plot)
        # Use shell() function with f-string
        shell(f"python {input.script} --serialized_data_path '{input.serialized_data}' --processed_data_path '{input.processed_data}' --output_dir '{output_dir}' --plot_dir '{plot_dir}' --model {model}")

rule evaluate_llm_llama_3_2_1b:
    input:
        script="code/5_evaluate_LLM.py",
        serialized_data="output_data/serialized_data.txt",
        processed_data="output_data/processed_data.parquet"
    output:
        results="output_data/llm_llama_3_2_1b_results.csv",
        plot="output_plots/llm_llama_3_2_1b_roc_curve.png"
    threads: 1
    run:
        model = "llama_3_2_1b"
        # Calculate directories within the run block
        output_dir = os.path.dirname(output.results)
        plot_dir = os.path.dirname(output.plot)
        # Use shell() function with f-string
        shell(f"python {input.script} --serialized_data_path '{input.serialized_data}' --processed_data_path '{input.processed_data}' --output_dir '{output_dir}' --plot_dir '{plot_dir}' --model {model}")

rule aggregate_llm_results:
    input:
        qwen0_5_results="output_data/llm_qwen_0_5b_results.csv",
        qwen1_5_results="output_data/llm_qwen_1_5b_results.csv",
        llama_results="output_data/llm_llama_3_2_1b_results.csv"
    output:
        aggregated_results="output_data/llm_evaluation_results.csv"
    threads: 1
    run:
        # Simple aggregation: concatenate the CSVs. 
        # A more sophisticated script could be used here for complex aggregation.
        with open(output.aggregated_results, 'w') as outfile:
            first_file = True
            for fname in [input.qwen0_5_results, input.qwen1_5_results, input.llama_results]:
                with open(fname, 'r') as infile:
                    if first_file:
                        outfile.write(infile.read())
                        first_file = False
                    else:
                        next(infile) # Skip header for subsequent files
                        outfile.write(infile.read())