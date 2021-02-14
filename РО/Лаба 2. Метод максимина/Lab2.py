import sys
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris


# кластеризатор
class MaxminMethod:
    def __init__(self, X_set):
        self.X_set = X_set
        self.clusters = []
        # метрика (евклидова)
        self.metric_func = lambda x, y: np.sqrt(((x - y) ** 2).sum())

    # поулчаем кластеры с объектами
    def get_clusters(self):
        result = {}
        for cluster in self.clusters:
            result[cluster['obj_prototype_idx']] = []
        for obj_idx in self.X_set.index:
            min_dist = sys.float_info.max
            obj_prototype_idx = -1
            for cluster in self.clusters:
                dist = cluster['dist'][obj_idx]
                if dist < min_dist:
                    min_dist = cluster['dist'][obj_idx]
                    obj_prototype_idx = cluster['obj_prototype_idx']
            result[obj_prototype_idx].append(obj_idx)
        return result

    # запуск алгоритма
    def start(self):
        self.clusters.clear()
        self.clusters.append({
            'obj_prototype_idx': 0,
            'dist': self.get_series_of_dist_to_obj(self.X_set.loc[0])
        })
        while True:
            prototype_obj_idx, max_dist = self.find_next_possible_prototype()
            if max_dist <= self.get_average_dist_of_clusters():
                break
            self.clusters.append({
                'obj_prototype_idx': prototype_obj_idx,
                'dist': self.get_series_of_dist_to_obj(self.X_set.loc[prototype_obj_idx])
            })

    # находим половину среднего арифметического всех расстояний между прототипами
    def get_average_dist_of_clusters(self):
        sum = 0.0
        count = 0
        for i in range(0, len(self.clusters)):
            for j in range(i + 1, len(self.clusters)):
                obj_prototype_idx = self.clusters[j]['obj_prototype_idx']
                sum = sum + self.clusters[i]['dist'][obj_prototype_idx]
                count = count + 1
        if count == 0:
            return 0.0
        return sum / (2 * count)

    # найти следующий потенциальный прототип
    def find_next_possible_prototype(self):
        max_dist = 0.0
        prototype_obj_idx = -1
        for obj_idx in self.X_set.index:
            min_dist = sys.float_info.max
            for cluster in self.clusters:
                dist = cluster['dist'][obj_idx]
                if dist < min_dist:
                    min_dist = cluster['dist'][obj_idx]
            if min_dist > max_dist:
                max_dist = min_dist
                prototype_obj_idx = obj_idx
        return prototype_obj_idx, max_dist

    # поулчить серию расстояний от всех объектов до данного
    def get_series_of_dist_to_obj(self, obj):
        return self.X_set.apply(lambda row: self.metric_func(row, obj), axis=1)


def main():
    # датасет
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)

    # кластеризация
    maxmin_method = MaxminMethod(X)
    maxmin_method.start()
    clusters = maxmin_method.get_clusters()
    print('end')


if __name__ == "__main__":
    main()