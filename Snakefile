import os

configfile: "config.yaml"

rule all:
    input:
        log_summary="output_data/baseline_logistic_summary.txt",
        log_processed="output_data/baseline_logistic_processed.parquet",
        log_plot="output_plots/baseline_logistic_roc_curve.png",
        oasis_plot="output_plots/baseline_oasis_roc_curve.png"

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
        config["inputdata_path"] + "data.parquet"
    threads: 1
    shell:
        "python {input.script} --config '{input.config}' --output '{output}'"

rule baseline_logistic:
    input:
        script="code/1_baseline_logistic.py",
        data=config["inputdata_path"] + "data.parquet"
    output:
        summary="output_data/baseline_logistic_summary.txt",
        processed="output_data/baseline_logistic_processed.parquet",
        plot="output_plots/baseline_logistic_roc_curve.png"
    threads: 1
    run:
        # Calculate directories within the run block
        output_dir = os.path.dirname(output.summary)
        plot_dir = os.path.dirname(output.plot)
        # Use shell() function with f-string
        shell(f"python {input.script} --input '{input.data}' --output_dir '{output_dir}' --plot_dir '{plot_dir}'")

rule baseline_oasis:
    input:
        script="code/1_baseline_oasis.py",
        data=config["inputdata_path"] + "data.parquet"
    output:
        plot="output_plots/baseline_oasis_roc_curve.png"
    threads: 1
    run:
        # Calculate directory within the run block
        plot_dir = os.path.dirname(output.plot)
        # Use shell() function with f-string
        shell(f"python {input.script} --input '{input.data}' --plot_dir '{plot_dir}'")