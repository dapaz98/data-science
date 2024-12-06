from pandas import read_csv, DataFrame
from data_profiling.data_dimensionality import dimensionality_process 
from data_profiling.data_distribution import distribution_process
from data_preparation.pre_processing import process_normalization
from data_preparation.dataset_evaluation import teste
from pathlib import Path
from models.process_models import process
def main():

    filename  = "/home/dapaz98/Documents/university/data-science/project/datasets/raw/financial_distress.csv"
    file_tag  = "financial_distress"
    index_col = "Company"
    financial_distress_data: DataFrame = read_csv(filename, index_col=index_col, na_values="")
    # print(type(financial_distress_data))
    # dimensionality_process(financial_distress_data, file_tag)
    # distribution_process(financial_distress_data, file_tag)
    # process_normalization(financial_distress_data, file_tag)
    # teste()

    process(file_tag)

if __name__ == '__main__':
    main()
