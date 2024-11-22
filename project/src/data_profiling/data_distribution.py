from matplotlib.pyplot import savefig, show, figure, subplots
from numpy import ndarray, log
from matplotlib.figure import Figure
from charts_config.dslabs_functions import define_grid, HEIGHT, get_variable_types, plot_multibar_chart
from charts_config.dslabs_functions import plot_bar_chart,plot_multibar_chart, set_chart_labels, plot_multiline_chart
from pandas import Series, DataFrame
from scipy.stats import norm, expon, lognorm
from matplotlib.axes import Axes

NR_STDEV: int = 2
IQR_FACTOR: float = 1.5
DIR: str = "/home/dapaz98/Documents/university/data-science/project/docs/data_distribution"

def save_boxplot(data, numeric, file_tag):

    if [] != numeric:
        data[numeric].boxplot(rot=45)
        savefig(f"{DIR}/{file_tag}_global_boxplot.png")
        
        rows: int
        cols: int
        rows, cols = define_grid(len(numeric))
        fig: Figure
        axs: ndarray
        fig, axs = subplots(
            rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
        )
        i, j = 0, 0
        for n in range(len(numeric)):
            axs[i, j].set_title("Boxplot for %s" % numeric[n])
            axs[i, j].boxplot(data[numeric[n]].dropna().values)
            i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
        savefig(f"{DIR}/{file_tag}_single_boxplots.png")
        
    else:
        print("There are no numeric variables.")


def save_histogram(data, numeric, file_tag):

    if [] != numeric:
        
        rows: int
        cols: int
        rows, cols = define_grid(len(numeric))
        fig: Figure
        axs: ndarray

        # Single Histogram
        fig, axs = subplots(
            rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
        )

        i, j = 0, 0
        for n in range(len(numeric)):
            set_chart_labels(
                axs[i, j],
                title=f"Histogram for {numeric[n]}",
                xlabel=numeric[n],
                ylabel="nr records",
            )
            axs[i, j].hist(data[numeric[n]].dropna().values, "auto")
            i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
        savefig(f"{DIR}/{file_tag}_single_histograms_numeric.png")
        
        i, j = 0, 0
        for n in range(len(numeric)):
            histogram_with_distributions(axs[i, j], data[numeric[n]].dropna(), numeric[n])
            i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
        savefig(f"images/{file_tag}_histogram_numeric_distribution.png")
        show()

    else:
        print("There are no numeric variables.")

def determine_outlier_thresholds_for_var(
    summary5: Series, std_based: bool = True, threshold: float = NR_STDEV
) -> tuple[float, float]:
    top: float = 0
    bottom: float = 0
    if std_based:
        std: float = threshold * summary5["std"]
        top = summary5["mean"] + std
        bottom = summary5["mean"] - std
    else:
        iqr: float = threshold * (summary5["75%"] - summary5["25%"])
        top = summary5["75%"] + iqr
        bottom = summary5["25%"] - iqr

    return top, bottom


def count_outliers(
    data: DataFrame,
    numeric: list[str],
    nrstdev: int = NR_STDEV,
    iqrfactor: float = IQR_FACTOR,
) -> dict:
    outliers_iqr: list = []
    outliers_stdev: list = []
    summary5: DataFrame = data[numeric].describe()

    for var in numeric:
        top: float
        bottom: float
        top, bottom = determine_outlier_thresholds_for_var(
            summary5[var], std_based=True, threshold=nrstdev
        )
        outliers_stdev += [
            data[data[var] > top].count()[var] + data[data[var] < bottom].count()[var]
        ]

        top, bottom = determine_outlier_thresholds_for_var(
            summary5[var], std_based=False, threshold=iqrfactor
        )
        outliers_iqr += [
            data[data[var] > top].count()[var] + data[data[var] < bottom].count()[var]
        ]

    return {"iqr": outliers_iqr, "stdev": outliers_stdev}

def identify_outliers(data, numeric, file_tag):
    if [] != numeric:
        outliers: dict[str, int] = count_outliers(data, numeric)
        figure(figsize=(12, HEIGHT))
        plot_multibar_chart(
            numeric,
            outliers,
            title="Nr of standard outliers per variable",
            xlabel="variables",
            ylabel="nr outliers",
            percentage=False,
        )
        savefig(f"{DIR}/{file_tag}_outliers_standard.png")
        show()
    else:
        print("There are no numeric variables.")

def compute_known_distributions(x_values: list) -> dict:
    distributions = dict()
    # Gaussian
    mean, sigma = norm.fit(x_values)
    distributions["Normal(%.1f,%.2f)" % (mean, sigma)] = norm.pdf(x_values, mean, sigma)
    # Exponential
    loc, scale = expon.fit(x_values)
    distributions["Exp(%.2f)" % (1 / scale)] = expon.pdf(x_values, loc, scale)
    # LogNorm
    sigma, loc, scale = lognorm.fit(x_values)
    distributions["LogNor(%.1f,%.2f)" % (log(scale), sigma)] = lognorm.pdf(
        x_values, sigma, loc, scale
    )
    return distributions


def histogram_with_distributions(ax: Axes, series: Series, var: str):
    values: list = series.sort_values().to_list()
    ax.hist(values, 20, density=True)
    distributions: dict = compute_known_distributions(values)
    plot_multiline_chart(
        values,
        distributions,
        ax=ax,
        title="Best fit for %s" % var,
        xlabel=var,
        ylabel="",
    )

def differente_types(variables_types,data, file_tag):
    
    symbolic: list[str] = variables_types["symbolic"] + variables_types["binary"]

    if [] != symbolic:
        rows, cols = define_grid(len(symbolic))
        fig, axs = subplots(
            rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
        )
        i, j = 0, 0
        for n in range(len(symbolic)):
            counts: Series = data[symbolic[n]].value_counts()
            plot_bar_chart(
                counts.index.to_list(),
                counts.to_list(),
                ax=axs[i, j],
                title="Histogram for %s" % symbolic[n],
                xlabel=symbolic[n],
                ylabel="nr records",
                percentage=False,
            )
            i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
        savefig(f"{DIR}/{file_tag}_histograms_symbolic.png")
        show()
    else:
        print("There are no symbolic variables.")

def class_distribution(data,file_tag):
    target = "CLASS"

    values: Series = data[target].value_counts()
    print(values)
    
    figure(figsize=(4, 2))
    plot_bar_chart(
        values.index.to_list(),
        values.to_list(),
        title=f"Target distribution (target={target})",
    )
    savefig(f"{DIR}/{file_tag}_class_distribution.png")

# TODO: Call the function to save the images to the process
def distribution_process(data, file_tag):
    variables_types: dict[str, list] = get_variable_types(data)
    numeric: list[str] = variables_types["numeric"]

    save_boxplot(data, numeric, file_tag)
    identify_outliers(data, numeric, file_tag)
    # save_histogram(data, numeric, file_tag)
    
    differente_types(variables_types,data, file_tag)
    class_distribution(data, file_tag)