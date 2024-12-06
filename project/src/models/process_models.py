import os 
from charts_config.dslabs_functions import read_train_test_from_files
from models.naive_bayes import nb_process
from models.knn import knn_process
from models.perceptron import perceptron_process
from models.random_forests import rf_process

DIR_EVAL = "/home/dapaz98/Documents/university/data-science/project/datasets/evaluation_dataset/financial_distress_oversampling"
DIR_SAVE_FIG = "/home/dapaz98/Documents/university/data-science/project/docs/models_approach"

def process(target):
    train_filename = os.path.join(DIR_EVAL, "train.csv")
    test_filename = os.path.join(DIR_EVAL, "test.csv")

    eval_metric = "accuracy"
    trnX, tstX, trnY, tstY, labels, vars = read_train_test_from_files(
    train_filename, test_filename, target)
    
    print(f"Train#={len(trnX)} Test#={len(tstX)}")
    print(f"Labels={labels}")

    # nb_process(trnX, trnY, tstX, tstY, eval_metric, labels, target)
    # knn_process(trnX, trnY, tstX, tstY, eval_metric, labels, target)
    # perceptron_process(trnX, trnY, tstX, tstY, eval_metric, labels, target)
    rf_process(trnX, trnY, tstX, tstY, eval_metric, labels, target, vars)