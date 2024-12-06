from numpy import ndarray, array
from imblearn.over_sampling import SMOTE
from pandas import DataFrame, Series, concat
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from charts_config.dslabs_functions import (
    NR_STDEV,
    get_variable_types,
    determine_outlier_thresholds_for_var,
    plot_bar_chart
)
import os

DIR = "/home/dapaz98/Documents/university/data-science/project/datasets/processed"
DIR_EVAL = "/home/dapaz98/Documents/university/data-science/project/datasets/evaluation_dataset"
TARGET_VARIABLE = "financial_distress"
CLASS_VARIABLE = "CLASS"

def normalize_target_variabel(data):
    data[TARGET_VARIABLE] = data[TARGET_VARIABLE].apply(lambda x: 0 if x >= -0.50 else 1)
    return data

def categorization_variable(data):
    data['CLASS'] = data['CLASS'].astype(bool)
    data['financial_distress'] = data['financial_distress'].astype(bool)

    return data

def replacing_outliers(data, file_tag):
    numeric_vars: list[str] = get_variable_types(data)["numeric"]

    if [] != numeric_vars:
        summary5: DataFrame = data[numeric_vars].describe()
        for var in numeric_vars:
            top, bottom = determine_outlier_thresholds_for_var(summary5[var])
            median: float = data[var].median()
            data[var] = data[var].apply(lambda x: median if x > top or x < bottom else x)

        data.to_csv(f"{DIR}/{file_tag}_replacing_outliers.csv", index=True)
        print("Data after replacing outliers:", data.shape)
    else:
        print("There are no numeric variables")
    return data

def generate_target_data(data):
    target_data = data[[TARGET_VARIABLE, CLASS_VARIABLE]].copy()
    return target_data

def standard_scaler(data, file_tag):
    remaining_vars = [col for col in data.columns if col not in [TARGET_VARIABLE, CLASS_VARIABLE]]

    target_data = generate_target_data(data)
    data.drop(columns=[TARGET_VARIABLE, CLASS_VARIABLE], inplace=True)


    transf: StandardScaler = StandardScaler(with_mean=True, with_std=True, copy=True).fit(
        data
    )

    df_zscore = DataFrame(transf.transform(data),columns=remaining_vars, index=data.index)
    df_zscore[[TARGET_VARIABLE, CLASS_VARIABLE]] = target_data
    full_file_tag = file_tag+ "_scaled_zscore"
    df_zscore.to_csv(f"{DIR}/{full_file_tag}.csv")
    split_dataset(df_zscore, full_file_tag)

def min_max_scaler(data, file_tag):
    remaining_vars = [col for col in data.columns if col not in [TARGET_VARIABLE, CLASS_VARIABLE]]

    target_data = generate_target_data(data)
    data.drop(columns=[TARGET_VARIABLE, CLASS_VARIABLE], inplace=True)

    transf: MinMaxScaler = MinMaxScaler(feature_range=(-1, 2), copy=True).fit(data)
    print("PASSOU")
    df_minmax = DataFrame(transf.transform(data), columns=remaining_vars, index=data.index)
    df_minmax[[TARGET_VARIABLE, CLASS_VARIABLE]] = target_data
    full_file_tag = file_tag+ "_scaled_minmax"
    df_minmax.to_csv(f"{DIR}/{full_file_tag}.csv", index="id")
    split_dataset(df_minmax, full_file_tag)

def data_balancing(data, file_tag):

    target_count: Series = data[TARGET_VARIABLE].value_counts()

    positive_class = target_count.idxmin()
    negative_class = target_count.idxmax()

    print("Minority class=", positive_class, ":", target_count[positive_class])
    print("Majority class=", negative_class, ":", target_count[negative_class])
    print(
        "Proportion:",
        round(target_count[positive_class] / target_count[negative_class], 2),
        ": 1",
    )
    values: dict[str, list] = {
        "Original": [target_count[positive_class], target_count[negative_class]]
    }

    df_positives: Series = data[data[TARGET_VARIABLE] == positive_class]
    df_negatives: Series = data[data[TARGET_VARIABLE] == negative_class]

    undersampling(df_positives, df_negatives, file_tag, positive_class, negative_class)
    oversampling(df_positives, df_negatives, file_tag, positive_class, negative_class)
    smote(data, file_tag, positive_class, negative_class)

def undersampling(df_positives, df_negatives, file_tag, positive_class, negative_class):
    df_neg_sample: DataFrame = DataFrame(df_negatives.sample(len(df_positives)))
    df_under: DataFrame = concat([df_positives, df_neg_sample], axis=0)
    full_file_tag = file_tag+ "_undersampling"
    df_under.to_csv(f"{DIR}/{full_file_tag}.csv", index=False)
    split_dataset(df_under, full_file_tag)
    print("Minority class=", positive_class, ":", len(df_positives))
    print("Majority class=", negative_class, ":", len(df_neg_sample))
    print("Proportion:", round(len(df_positives) / len(df_neg_sample), 2), ": 1")

def oversampling(df_positives, df_negatives, file_tag, positive_class, negative_class):
    df_pos_sample: DataFrame = DataFrame(
        df_positives.sample(len(df_negatives), replace=True)
    )
    df_over: DataFrame = concat([df_pos_sample, df_negatives], axis=0)
    full_file_tag = file_tag+ "_oversampling"
    df_over.to_csv(f"{DIR}/{full_file_tag}.csv", index=False)
    split_dataset(df_over, full_file_tag)

    print("Minority class=", positive_class, ":", len(df_pos_sample))
    print("Majority class=", negative_class, ":", len(df_negatives))
    print("Proportion:", round(len(df_pos_sample) / len(df_negatives), 2), ": 1")

def smote(data, file_tag, positive_class, negative_class):
    RANDOM_STATE = 42

    smote: SMOTE = SMOTE(sampling_strategy="minority", random_state=RANDOM_STATE)
    y = data.pop(TARGET_VARIABLE).values
    X: ndarray = data.values
    smote_X, smote_y = smote.fit_resample(X, y)
    df_smote: DataFrame = concat([DataFrame(smote_X), DataFrame(smote_y)], axis=1)
    df_smote.columns = list(data.columns) + [TARGET_VARIABLE]
    full_file_tag = file_tag+ "_smote"
    df_smote.to_csv(f"{DIR}/{full_file_tag}.csv", index=False)
    split_dataset(df_smote, full_file_tag)

    smote_target_count: Series = Series(smote_y).value_counts()
    print("Minority class=", positive_class, ":", smote_target_count[positive_class])
    print("Majority class=", negative_class, ":", smote_target_count[negative_class])
    print(
        "Proportion:",
        round(smote_target_count[positive_class] / smote_target_count[negative_class], 2),
        ": 1",
    )

def split_dataset(data, file_tag):
    full_dir = os.path.join(DIR_EVAL, file_tag)
    if not os.path.exists(full_dir):  # Verifica se o diretório existe
        os.makedirs(full_dir)
    labels: list = list(data[TARGET_VARIABLE].unique())
    labels.sort()
    positive: int = 1
    negative: int = 0
    values: dict[str, list[int]] = {
        "Original": [
            len(data[data[TARGET_VARIABLE] == negative]),
            len(data[data[TARGET_VARIABLE] == positive]),
        ]
    }
    y: array = data.pop(TARGET_VARIABLE).to_list()
    X: ndarray = data.values
    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)
    train: DataFrame = concat(
        [DataFrame(trnX, columns=data.columns), DataFrame(trnY, columns=[TARGET_VARIABLE])], axis=1
    )
    train.to_csv(f"{full_dir}/train.csv", index=False)
    print("passou")
    test: DataFrame = concat(
        [DataFrame(tstX, columns=data.columns), DataFrame(tstY, columns=[TARGET_VARIABLE])], axis=1
    )
    test.to_csv(f"{full_dir}/test.csv", index=False)

def process_normalization(data, file_tag):
    data = normalize_target_variabel(data)
    data = categorization_variable(data)
    data = replacing_outliers(data, file_tag)
    # standard_scaler(data, file_tag)
    # min_max_scaler(data, file_tag)
    data_balancing(data, file_tag)