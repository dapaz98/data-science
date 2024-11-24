from pandas import read_csv, DataFrame
from data_profiling.data_dimensionality import dimensionality_process 
from data_profiling.data_distribution import distribution_process
def main():

    filename = "/home/dapaz98/Documents/university/data-science/project/datasets/raw/financial_distress.csv"
    file_tag = "financial_distress"
    financial_distress_data: DataFrame = read_csv(filename, na_values="", index_col="Company")
    # dimensionality_process(financial_distress_data, file_tag)
    distribution_process(financial_distress_data, file_tag)

if __name__ == '__main__':
    main()
