import sys
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris


# кластеризатор
class KMeansMethod:
    def __init__(self, X_set, cluster_count):
        self.X_set = X_set
        self.cluster_count = cluster_count
        self.clusters = []
        # метрика (евклидова)
        self.metric_func = lambda x, y: ((x - y) ** 2).sum()
        self.delta_eps = 0.01 * 0.01

    # поулчаем кластеры с объектами
    def get_clusters(self, indexes=True):
        cluster_objs = {}
        for obj_idx, obj in self.X_set.iterrows():
            min_dist = sys.float_info.max
            cluster_idx = -1
            for (i, prototype_obj) in enumerate(self.clusters):
                dist = self.metric_func(obj, prototype_obj)
                if dist < min_dist:
                    min_dist = dist
                    cluster_idx = i
            if indexes:
                cluster_objs.setdefault(cluster_idx, []).append(obj_idx)
            else:
                cluster_objs.setdefault(cluster_idx, []).append(obj)
        return cluster_objs

    # запуск алгоритма
    def start(self):
        self.clusters.clear()
        for i in range(0, self.cluster_count):
            self.clusters.append(self.X_set.loc[i])

        while True:
            cluster_objs = self.get_clusters(False)
            end = True
            for cluster_idx, objs in cluster_objs.items():
                df = pd.DataFrame(objs)
                mean_val = df.mean()
                if self.metric_func(self.clusters[cluster_idx], mean_val) >= self.delta_eps:
                    self.clusters[cluster_idx] = mean_val
                    end = False
            if end:
                break


def main():
    # датасет
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)

    # кластеризация
    k_means_method = KMeansMethod(X, 3)
    k_means_method.start()
    clusters = k_means_method.get_clusters()
    print('end')


if __name__ == "__main__":
    main()