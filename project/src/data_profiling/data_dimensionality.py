from matplotlib.pyplot import figure, savefig, show
from charts_config.dslabs_functions import plot_bar_chart, get_variable_types

def analyze_records(data, file_tag):
    figure(figsize=(6, 4))
    values: dict[str, int] = {"nr records": data.shape[0], "nr variables": data.shape[1]}
    plot_bar_chart(
        list(values.keys()), list(values.values()), title="Nr of records vs nr variables"
    )
    savefig(f"/home/dapaz98/Documents/university/data-science/project/docs/data_dimensionality/{file_tag}_records_variables.png")
    show()
    print()

def save_missing_values(data, file_tag):
    mv: dict[str, int] = {}
    for var in data.columns:
        nr: int = data[var].isna().sum()
        if nr > 0:
            mv[var] = nr

    figure(figsize=(6, 4))

    plot_bar_chart(
        list(mv.keys()),
        list(mv.values()),
        title="Nr of missing values per variable",
        xlabel="variables",
        ylabel="nr missing values",
    )
    savefig(f"/home/dapaz98/Documents/university/data-science/project/docs/data_dimensionality/{file_tag}_mv.png")
    show()

def save_variables_type(data, file_tag):
    variable_types: dict[str, list] = get_variable_types(data)
    print(variable_types)
    counts: dict[str, int] = {}
    for tp in variable_types.keys():
        counts[tp] = len(variable_types[tp])

    figure(figsize=(6, 4))

    plot_bar_chart(
        list(counts.keys()), list(counts.values()), title="Nr of variables per type"
    )
    savefig(f"/home/dapaz98/Documents/university/data-science/project/docs/data_dimensionality/{file_tag}_variable_types.png")
    show()

# TODO: Call the function to save the images to the process
def dimensionality_process(data, file_tag):
    analyze_records(data, file_tag)
    save_missing_values(data, file_tag)
    save_variables_type(data, file_tag)






