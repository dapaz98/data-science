from typing import Literal
from numpy import array, ndarray
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.pyplot import figure, savefig, show
from charts_config.dslabs_functions import CLASS_EVAL_METRICS, DELTA_IMPROVE, plot_multiline_chart
from charts_config.dslabs_functions import read_train_test_from_files, plot_evaluation_results

DIR_SAVE_FIG = "/home/dapaz98/Documents/university/data-science/project/docs/models_approach"
"""
    manhattan: Soma das diferenças absolutas entre as coordenadas. Também chamada de distância "L1".
    euclidean: Raiz quadrada da soma dos quadrados das diferenças (distância "L2").
    chebyshev: A maior diferença absoluta entre as coordenadas.
"""

def knn_study(
        trnX: ndarray, trnY: array, tstX: ndarray, tstY: array, k_max: int=19, lag: int=2, metric='accuracy'
        ) -> tuple[KNeighborsClassifier | None, dict]:
    dist: list[Literal['manhattan', 'euclidean', 'chebyshev']] = ['manhattan', 'euclidean', 'chebyshev']

    kvalues: list[int] = [i for i in range(1, k_max+1, lag)]
    best_model: KNeighborsClassifier | None = None
    best_params: dict = {'name': 'KNN', 'metric': metric, 'params': ()}
    best_performance: float = 0.0

    values: dict[str, list] = {}
    for d in dist:
        y_tst_values: list = []
        for k in kvalues:
            clf = KNeighborsClassifier(n_neighbors=k, metric=d)
            clf.fit(trnX, trnY)
            prdY: array = clf.predict(tstX)
            eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
            y_tst_values.append(eval)
            if eval - best_performance > DELTA_IMPROVE:
                best_performance: float = eval
                best_params['params'] = (k, d)
                best_model = clf
            # print(f'KNN {d} k={k}')
        values[d] = y_tst_values
    
    print(f'KNN best with k={best_params['params'][0]} and {best_params['params'][1]}')
    plot_multiline_chart(kvalues, values, title=f'KNN Models ({metric})', xlabel='k', ylabel=metric, percentage=True)

    return best_model, best_params

def overfitting(trnX, trnY, tstX, tstY, eval_metric, target):
    distances = ["manhattan", "euclidean", "chebyshev"]  # Lista de métricas de distância
    K_MAX = 25
    kvalues: list[int] = [i for i in range(1, K_MAX, 2)]
    acc_metric: str = "accuracy"
    
    for distance in distances:  # Itera sobre cada métrica de distância
        y_tst_values: list = []
        y_trn_values: list = []
        
        for k in kvalues:  # Itera sobre os valores de k
            clf = KNeighborsClassifier(n_neighbors=k, metric=distance)
            clf.fit(trnX, trnY)
            prd_tst_Y: array = clf.predict(tstX)
            prd_trn_Y: array = clf.predict(trnX)
            y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
            y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))
        
        # Gera o gráfico para a métrica de distância atual
        figure()
        plot_multiline_chart(
            kvalues,
            {"Train": y_trn_values, "Test": y_tst_values},
            title=f"KNN overfitting study for {distance}",
            xlabel="K",
            ylabel=str(eval_metric),
            percentage=True,
        )
        savefig(f"{DIR_SAVE_FIG}/{target}_knn_overfitting_{distance}.png")
        show()
    
def knn_process(trnX, trnY, tstX, tstY, eval_metric, labels, target):
    figure()
    best_model, params = knn_study(trnX, trnY, tstX, tstY, k_max=25, metric=eval_metric)
    
    savefig(f'{DIR_SAVE_FIG}/{target}_knn_{eval_metric}_study.png')
    
    # Performance
    prd_trn: array = best_model.predict(trnX)
    prd_tst: array = best_model.predict(tstX)
    figure()
    plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
    savefig(f'{DIR_SAVE_FIG}/{target}_knn_{params["name"]}_best_{params["metric"]}_eval.png')
    show()

    overfitting(trnX, trnY, tstX, tstY, eval_metric, target)