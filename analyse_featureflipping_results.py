"""
Read the results, collect them and save a summarisation.
"""
import os
import pickle
import pandas as pd
import numpy as np
import csv


def generate_latex_table(data):
    latex_table = "\\begin{table*}[t]\n"
    latex_table += "\\centering\n"
    latex_table += "\\begin{tabular}{lccccccccccc}\n"

    # Add "Dataset" before the column names and make all column names bold
    headers = data[0]
    headers = ["\\textbf{Dataset}"] + ["\\textbf{" + col + "}" for col in headers[1:]]

    # Add column headers
    latex_table += " & ".join(headers) + " \\\\\n"
    latex_table += "\\hline\n"

    for idx, row in enumerate(data[1:], start=1):
        rounded_row = []
        min_value = float('inf')  # Initialize with a high value
        min_index = -1
        second_min_value = float('inf')  # Initialize with a high value
        second_min_index = -1

        for i, value in enumerate(row):
            if value.replace('.', '', 1).isdigit():
                float_value = float(value)
                if float_value < min_value:
                    second_min_value = min_value
                    second_min_index = min_index
                    min_value = float_value
                    min_index = i
                elif float_value < second_min_value:
                    second_min_value = float_value
                    second_min_index = i
                rounded_row.append("{:.3f}".format(float_value))
            else:
                rounded_row.append(value)

        # Highlight the smallest value in the row as bold and the second-smallest value as underlined
        if min_index >= 0:
            rounded_row[min_index] = "\\textbf{" + rounded_row[min_index] + "}"
        if second_min_index >= 0:
            rounded_row[second_min_index] = "\\underline{" + rounded_row[second_min_index] + "}"

        latex_table += row[0] + " & " + " & ".join(rounded_row[1:]) + " \\\\\n"

        # Add a horizontal line between the last and second-to-last rows
        if idx == len(data) - 2:
            latex_table += "\\hline\n"

    latex_table += "\\end{tabular}\n"
    latex_table += "\\caption{AUC results of the pixelflipping experiment. Lower AUC values are better.}\n"
    latex_table += "\\label{table:pf}\n"
    latex_table += "\\end{table*}"

    return latex_table

def generate_latex_table(data, df_stds):
    latex_table = "\\begin{table*}[t]\n"
    latex_table += "\\centering\n"
    latex_table += "\\begin{tabular}{l"+"c"*(len(data[0]))+"}\n"

    # Add "Dataset" before the column names and make all column names bold
    headers = data[0]
    headers = ["\\textbf{Dataset}"] + ["\\textbf{" + col + "}" for col in headers[1:]]

    # Add column headers
    latex_table += " & ".join(headers) + " \\\\\n"
    latex_table += "\\hline\n"

    for idx, row in enumerate(data[1:], start=1):# If stds is shorter than data, pad with zeros
        formatted_row = []
        min_value = float('inf')
        min_index = -1
        second_min_value = float('inf')
        second_min_index = -1

        for i, value in enumerate(row):
            if value.replace('.', '', 1).isdigit():
                float_value = float(value)

                # Find the smallest and second smallest values
                if float_value < min_value:
                    second_min_value = min_value
                    second_min_index = min_index
                    min_value = float_value
                    min_index = i
                elif float_value < second_min_value:
                    second_min_value = float_value
                    second_min_index = i

                # Format value with standard deviation
                formatted_value = "{:.3f}".format(float_value)
                formatted_row.append(formatted_value)
            else:
                formatted_row.append(value)

        # Highlight the smallest value in the row as bold and the second-smallest value as underlined
        if min_index >= 0:
            formatted_row[min_index] = "\\textbf{" + formatted_row[min_index] + "}"
        if second_min_index >= 0:
            formatted_row[second_min_index] = "\\underline{" + formatted_row[second_min_index] + "}"

        # add to row the max std
        if not row[0] == "mean":
            formatted_row.append("{:.3f}".format(df_stds["max"][idx-1]))
        else:
            formatted_row.append("")

        # Add the row to the LaTeX table
        latex_table += row[0] + " & " + " & ".join(formatted_row[1:]) + " \\\\\n"

        # Add a horizontal line between the last and second-to-last rows
        if idx == len(data) - 2:
            latex_table += "\\hline\n"

    latex_table += "\\end{tabular}\n"
    latex_table += "\\caption{AUC results of the pixelflipping experiment. Lower AUC values are better.}\n"
    latex_table += "\\label{table:pf}\n"
    latex_table += "\\end{table*}"

    return latex_table


def format_value(val, bold_val, underline_val):
    """Format the value with LaTeX bold or underline as required."""
    if val == bold_val:
        return r"\textbf{" + f"{val:.3f}" + r"}"
    elif val == underline_val:
        return r"\underline{" + f"{val:.3f}" + r"}"
    else:
        return f"{val:.3f}"


def create_latex_table(df, std_df):
    """Create a LaTeX formatted table from the given DataFrame, including row names and sub-columns."""
    latex_table = []
    latex_table.append("\\begin{table*}[t]\n")
    latex_table.append("\\centering\n")
    latex_table.append("\\begin{tabular}{l" + "c" * (len(data[0])) + "}\n")
    latex_table.append("\\toprule\n")

    # Create header with sub-columns
    header = ["Dataset"]  # First column for row names
    sub_header = [""]  # This will hold the sub-column names

    for col in df.columns:
        # put a "/" infront of each "_"
        col = col.replace("_", "\\_")
        header.append(col)

    header.append("std (max)")

    latex_table.append(" & ".join(header) + r" \\")
    latex_table.append(" & ".join(sub_header) + r" \\")
    latex_table.append("\\midrule\n")
    # Add rows with values
    for index, row in df.iterrows():
        formatted_row = [index]  # Start with the row name

        min_row_value = row.min()
        min_row_value_col = row.idxmin()
        second_min_row_value = row[row != min_row_value].min()
        second_min_row_value_col = row[row != min_row_value].idxmin()

        for col in df.columns:
            val = row[col]

            if col == min_row_value_col:
                formatted_row.append(r"\textbf{" + f"{val:.3f}" + r"}")
            elif col == second_min_row_value_col:
                formatted_row.append(r"\underline{" + f"{val:.3f}" + r"}")
            else:
                formatted_row.append(f"{val:.3f}")

        if not index == "mean":
            formatted_row.append("{:.3f}".format(std_df.loc[index, "max"]))
        else:
            formatted_row.append("")

        latex_table.append(" & ".join(formatted_row) + r" \\")

    latex_table.append("\\end{tabular}\n")
    latex_table.append("\\caption{AUC results of the pixelflipping experiment. Lower AUC values are better.}\n")
    latex_table.append("\\label{table:pf}\n")
    latex_table.append("\\end{table*}")

    return "\n".join(latex_table)


if __name__ == "__main__":
    uncertainty_type = "MC_dropout"
    n_models = 10

    save_dir = "results/pixelflipping/{}/{}_models".format(uncertainty_type, n_models)

    # read all pkl files in the results folder
    pkl_files = [f for f in os.listdir(save_dir) if f.endswith(".pkl")]

    # read files into dictionary
    results_dict = {}
    for file in pkl_files:
        with open(save_dir+"/"+file, 'rb') as f:
            file_name = file.split(".")[0]
            #if file_name == "EPEX-FR":
                #continue
            results_dict[file_name] = pickle.load(f)

    # benchmark explanation names
    first_auc_dict = results_dict[list(results_dict.keys())[0]]["auc"]
    first_std_dict = results_dict[list(results_dict.keys())[0]]["auc_stds"]
    auc_keys = list(first_auc_dict.keys())
    std_keys = list(first_std_dict.keys())

    # make pandas dataframe to contain the auc values in the results, column names are auc keys
    auc_df = pd.DataFrame(columns=auc_keys)
    # init all dataset rows with 0 in all columns
    std_df = pd.DataFrame(columns=std_keys)

    # iterate through results_dict, collect auc values and save them in the dataframe
    for key in results_dict.keys():
        auc_dict = results_dict[key]["auc"]
        auc_df.loc[key] = auc_dict
    # add to the dataframe the mean auc value row
    auc_df.loc["mean"] = auc_df.mean()

    for key in results_dict.keys():
        n_samples = results_dict[key]["n_samples"]
        std_dict = results_dict[key]["auc_stds"]

        # std_mean_estimator is std_dict but each value is divided by the square root of the number of samples
        std_mean_estimator_dict = {k: v/np.sqrt(n_samples) for k, v in std_dict.items()}
        std_df.loc[key] = std_mean_estimator_dict

    explanation_columns = [
        "CovLRP_diag", "CovLRP_marg", "LRP",
        "CovGI_diag", "CovGI_marg", "GI",
        "CovIG_diag", "CovIG_marg", "IG",
        "CovShapley_diag", "CovShapley_marg", "Shapley",
        "Sensitivity", "LIME"]

    auc_df = auc_df[explanation_columns]

    auc_df.to_csv("results/auc_results.csv")

    csv_file = "results/auc_results.csv"
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        data = list(csv_reader)

    # get Cov method column names, i.e. the set of column names that start with "Cov", without the "_diag" and "_marg" suffixes and without the "Cov" prefix
    cov_methods = set([col.split('_')[0].split("Cov")[1] for col in data[0] if col.startswith("Cov")])
    # expand into column names, keeping the following order "LRP", "CovLRP_diag", "CovLRP_marg", etc.

    column_names = []
    for cov in cov_methods:
        column_names.append("Cov"+f"{cov}_diag")
        column_names.append("Cov"+f"{cov}_marg")
        column_names.append(cov)

    # then add the remaining columns
    column_names.extend([col for col in data[0] if col not in column_names])
    # drop ""
    column_names.remove("")
    # reorder the columns
    auc_df = auc_df[column_names]
    std_df = std_df[column_names]
    std_df["max"] = std_df.max(axis=1)

    # change the order of the columns such t

    # Generate the LaTeX table format
    latex_table = create_latex_table(auc_df, std_df)

    print(latex_table)

    print("Done")