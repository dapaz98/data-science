from numpy import array, ndarray
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from charts_config.dslabs_functions import (CLASS_EVAL_METRICS, DELTA_IMPROVE, 
                                            plot_bar_chart, plot_evaluation_results)
from matplotlib.pyplot import figure, savefig

DIR_SAVE_FIG = "/home/dapaz98/Documents/university/data-science/project/docs/models_approach"

"""
    MultinomialNB, we dont evaluate this approach because dont accept the negative values
"""
def naive_Bayes_study(
    target, trnX: ndarray, trnY: array, tstX: ndarray, tstY: array, metric: str = "accuracy"
) -> tuple:
    estimators: dict = {
        "GaussianNB": GaussianNB(),
        "BernoulliNB": BernoulliNB(),
    }
    xvalues: list = []
    yvalues: list = []
    best_model = None
    best_params: dict = {"name": "", "metric": metric, "params": ()}
    best_performance = 0
    for clf in estimators:
        xvalues.append(clf)
        estimators[clf].fit(trnX, trnY)
        prdY: array = estimators[clf].predict(tstX)
        eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
        if eval - best_performance > DELTA_IMPROVE:
            best_performance: float = eval
            best_params["name"] = clf
            best_params[metric] = eval
            best_model = estimators[clf]
        yvalues.append(eval)

    plot_bar_chart(
        xvalues,
        yvalues,
        title=f"Naive Bayes Models ({metric})",
        ylabel=metric,
        percentage=True,
    )
    savefig(f"{DIR_SAVE_FIG}/{target}_nb_{metric}_study.png")
    return best_model, best_params

def evaluation_models(best_model, params, target, trnX, trnY, tstX, tstY, eval_metric, labels):        
    prd_trn: array = best_model.predict(trnX)
    prd_tst: array = best_model.predict(tstX)
    figure()
    plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
    savefig(f'{DIR_SAVE_FIG}/{target}_{params["name"]}_best_{params["metric"]}_eval.png')
    
def nb_process(trnX, trnY, tstX, tstY, eval_metric, labels, target):
    best_model, params = naive_Bayes_study(eval_metric, target, trnX, trnY, tstX, tstY)
    evaluation_models(best_model, params, target, trnX, trnY, tstX, tstY, eval_metric, labels)