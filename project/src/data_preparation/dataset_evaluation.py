from numpy import ndarray 
from matplotlib.pyplot import figure, savefig
from charts_config.dslabs_functions import plot_multibar_chart, evaluate_approach, run_KNN
import os
from pandas import read_csv, DataFrame
# Exemplo de uso
DIR_PROCESSED = "/home/dapaz98/Documents/university/data-science/project/datasets/evaluation_dataset"
TARGET_VARIABLE = "financial_distress"  # Substitua pelo nome real da variável alvo
DIR_SAVE_FIG = "/home/dapaz98/Documents/university/data-science/project/docs/classification"

def knn_evaluate_approach(
    train: DataFrame, test: DataFrame, target: str = "class", metric: str = "accuracy"
) -> dict[str, list]:
    # Separate features and target
    trnY = train.pop(target).values
    trnX: ndarray = train.values
    tstY = test.pop(target).values
    tstX: ndarray = test.values

    # Evaluate using KNN
    eval_KNN: dict[str, float] = run_KNN(trnX, trnY, tstX, tstY, metric=metric)

    # Format the results for KNN only
    eval: dict[str, list] = {met: [eval_KNN[met]] for met in eval_KNN.keys()}
    return eval

def process_classification(base_dir, target_variable, dir_save_fig):
    """
    Processa todos os datasets de treino e teste em subdiretórios e gera gráficos de avaliação.

    :param base_dir: Diretório base contendo os subdiretórios com datasets
    :param target_variable: Variável alvo para classificação
    :param dir_save_fig: Diretório onde os gráficos serão salvos
    """
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)

        if os.path.isdir(subdir_path):
            print(f"Processando subdiretório: {subdir}")

            train_file = os.path.join(subdir_path, "train.csv")
            test_file = os.path.join(subdir_path, "test.csv")

            if os.path.exists(train_file) and os.path.exists(test_file):
                train = read_csv(train_file)
                test = read_csv(test_file)
                
                # Avaliação e geração de gráficos
                eval: dict[str, list] = knn_evaluate_approach(
                    train, test, target=target_variable, metric="recall"
                )
                
                # Configurar gráfico
                figure(figsize=(6, 4))
                plot_multibar_chart(
                    ["KNN"], eval, title=f"{subdir} evaluation", percentage=True
                )
                
                # Salvar gráfico
                os.makedirs(dir_save_fig, exist_ok=True)
                save_path = os.path.join(dir_save_fig, f"{subdir}_evaluation.png")
                savefig(save_path)
                print(f"Gráfico salvo em: {save_path}")

            else:
                print(f"Arquivos train.csv ou test.csv não encontrados em {subdir_path}")


def teste():
    process_classification(DIR_PROCESSED, TARGET_VARIABLE, DIR_SAVE_FIG)
