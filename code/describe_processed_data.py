import pandas as pd
from tableone import TableOne


# Load processed data
df = pd.read_parquet("./output_data/processed_data_basic.parquet")

# Print columns for inspection
print("Columns in processed_data_basic.parquet:")
print(df.columns.tolist())



# Use 'Global ICU Stay ID' as the source column and drop it after assignment
source_col = "Global ICU Stay ID"
df["source"] = (
    df[source_col]
    .astype(str)
    .str.lower()
    .str.startswith("mimic")
    .map({True: "MIMIC", False: "eICU"})
)
df = df.drop(columns=[source_col])

# Define categorical and continuous columns
categorical = [
    col
    for col in df.columns
    if df[col].dtype == "object" or df[col].nunique() < 10 and col != "source"
]
continuous = [
    col
    for col in df.columns
    if df[col].dtype != "object" and col not in categorical
]


# Create Table 1 stratified by source, without p-values
table1 = TableOne(
    df,
    columns=categorical + continuous,
    categorical=categorical,
    continuous=continuous,
    groupby="source",
    pval=False,
)

# Save as LaTeX
latex_path = "./output_data/tableone_processed_data_basic.tex"
with open(latex_path, "w") as f:
    f.write(table1.tabulate(tablefmt="latex"))
print(f"LaTeX table saved to {latex_path}")
