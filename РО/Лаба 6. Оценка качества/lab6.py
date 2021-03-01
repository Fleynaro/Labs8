import sys
import pandas as pd
import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sklearn.metrics as metrics


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


# классификатор
class KNearestNeighbors:
    def __init__(self, k):
        self.k = k
        self.metric_func = lambda x, y: ((x - y) ** 2).sum()

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, sample):
        indexes = self.X_train.apply(lambda row: self.metric_func(row, sample), axis=1).to_frame().sort_values(0).head(
            self.k).index
        return self.y_train[indexes].value_counts().idxmax()

    def predict_test_set(self, X_test):
        return X_test.apply(lambda row: self.predict(row), axis=1)


def add_bias_feature(a):
    a_extended = np.zeros((a.shape[0], a.shape[1] + 1))
    a_extended[:, :-1] = a
    a_extended[:, -1] = 1.
    return a_extended


# классификатор
class SVM:
    def __init__(self):
        self.w = None

    def train(self, X_train, y_train, epoches_count=1000, batch_size=20, alpha=0.1):
        # создаем вектор весов, расширяя пространство на 1
        self.w = torch.rand([X_train.shape[1] + 1, 1], requires_grad=True)

        # стохастический градиентный спуск
        optimizer = torch.optim.SGD([self.w], lr=0.001)
        for epoch in range(epoches_count):
            # перемешиваем индексы
            order = np.random.permutation(len(X_train))
            sum_loss = 0
            for start_index in range(0, len(X_train), batch_size):
                optimizer.zero_grad()
                # формируем батч
                batch_indexes = order[start_index:start_index + batch_size]
                X_batch = torch.tensor(add_bias_feature(X_train.iloc[batch_indexes].values).astype(np.float32))
                y_batch = torch.tensor(y_train.iloc[batch_indexes].values.astype(np.float32)).unsqueeze(1)

                # построение графа вычислений для функции потерь SVM
                loss_val = torch.clamp(1. - y_batch * X_batch @ self.w, min=0)
                loss_val = loss_val.mean() + alpha * self.w.t() @ self.w / 2.

                # нахождение градиента
                loss_val.backward()
                optimizer.step()
                sum_loss += float(loss_val)
            #predicted = self.predict_test_set(X_test)
            #accuracy = (y_test == predicted).mean()
            #print(f'accuracy = {accuracy}, sum_loss = {sum_loss}')

    def predict(self, sample):
        x = sample.values
        x = np.append(x, 1.)
        label = self.w.t() @ torch.tensor(x.astype(np.float32))
        return np.sign(label.item())

    def predict_test_set(self, X_test):
        return X_test.apply(lambda row: self.predict(row), axis=1)


def draw_roc(y_test, predicted):
    df = pd.DataFrame({'score': predicted, 'label': y_test}).sort_values(by='score', ascending=False)
    pos_labels = df[df.label >= 0.0]
    neg_labels = df[df.label < 0.0]
    x_grid = [i / len(neg_labels) for i in range(0, len(neg_labels) + 1)]
    x_grid.insert(0, -0.001)
    y_values = [0.0 for i in range(0, len(neg_labels) + 2)]

    i = 1
    cur_step_up = 0.0
    for idx, row in df.iterrows():
        if row[1] >= 0:
            cur_step_up += 1.0 / len(pos_labels)
        else:
            y_values[i] = cur_step_up
            i += 1
    y_values[i] = cur_step_up

    plt.plot(x_grid, y_values, marker='s')
    plt.plot([0.0, 1.0], [0.0, 1.0])
    plt.show()


def estimate_clf(y_test, predicted):
    TP = ((y_test == 1.0) & (predicted == 1.0)).sum()
    TN = ((y_test == -1.0) & (predicted == -1.0)).sum()
    FP = ((y_test == -1.0) & (predicted == 1.0)).sum()
    FN = ((y_test == 1.0) & (predicted == -1.0)).sum()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    MSE = (y_test - predicted).pow(2).mean()
    print(f'1) точность = {precision}, полнота = {recall}, F1-мера = {f1}')
    print(f'2) ROC-кривая')
    draw_roc(y_test, predicted)
    print(f'3) MSE = {MSE}')
    print(f'4) матрица неточности:\n{TN} | {FN}\n{FP} | {TP}')


def cross_validation(clf, X, y, k=5):
    order = np.random.permutation(X.index)
    part_size = len(X) // k
    total_accuracy = 0.0
    for start_index in range(0, len(order), part_size):
        train_indexes = np.hstack([order[:start_index], order[start_index + part_size:]])
        test_indexes = order[start_index:start_index + part_size]
        clf.train(X.loc[train_indexes], y[train_indexes])
        predicted = clf.predict_test_set(X.loc[test_indexes])
        accuracy = (y[test_indexes] == predicted).mean()
        total_accuracy += accuracy
    return total_accuracy / k


def to_df(clusters):
    result = None
    for k, v in clusters.items():
        df = X.loc[v].copy()
        df['cluster'] = k
        result = pd.concat([result, df])
    return result

# расстояние между кластерами (расстояние между ближайшими объектами)
def get_min_dist_between_clusters(df, c1, c2):
    cluster_objs1 = df[df.cluster == c1].drop('cluster', axis=1)
    cluster_objs2 = df[df.cluster == c2].drop('cluster', axis=1)
    return cluster_objs1.apply(
        lambda obj1: cluster_objs2.apply(
            lambda obj2: np.linalg.norm(obj1 - obj2), axis=1).min(), axis=1).min()


def CoefficientOfDetermination(df):
    all_objs = df.drop('cluster', axis=1)
    global_center = all_objs.mean()
    # глобальная изменчивость (SST = SSW + SSB)
    SST = all_objs.apply(
        lambda obj: ((obj - global_center) ** 2).sum(), axis=1).sum()
    # внутригрупповая изменчивость
    SSB = 0.0
    for cluster_idx in df.cluster.unique():
        cluster_objs = df[df.cluster == cluster_idx].drop('cluster', axis=1)
        cluster_center = cluster_objs.mean()
        SSB += cluster_objs.apply(
            lambda obj: ((obj - cluster_center) ** 2).sum(), axis=1).sum()
    return 1.0 - SSB / SST


def Silhouette(df):
    # найти ближайший кластер к объекту, ненаходящемуся в нем
    def get_nearest_cluster(obj):
        return df[df.cluster != obj.cluster].groupby('cluster').apply(
            lambda objs: np.linalg.norm(objs.mean()[:-1] - obj[:-1])).idxmin()
    # среднее расстояние от obj до других объектов из того же кластера
    def calc_a(obj):
        return df[df.cluster == obj.cluster].apply(
            lambda obj2: np.linalg.norm((obj[:-1] - obj2[:-1]).values), axis=1).mean()
    # среднее расстояние от obj до других объектов из другого кластера (ближайшего)
    def calc_b(obj, nearest_cluster):
        return df[df.cluster == nearest_cluster].apply(
            lambda obj2: np.linalg.norm((obj[:-1] - obj2[:-1]).values), axis=1).mean()
    # итоговая формула
    def score_per_obj(a, b):
        return (b - a) / max(a, b)
    return df.apply(
        lambda obj: score_per_obj(calc_a(obj), calc_b(obj, get_nearest_cluster(obj))), axis=1).mean()


# суть: делим самое минимальное межкластерное расстояние на самый максимальный кластерный диаметр
def DunnIndex(df):
    # самое минимальное межкластерное расстояние между кластерами
    clusters = df.cluster.unique()
    min_dist_between_clusters = pd.Series([
        get_min_dist_between_clusters(df, c1, c2) for c1 in clusters for c2 in clusters if c1 < c2]).min()
    # самый максимальный кластерный диаметр
    max_cluster_diameter = df.groupby('cluster').apply(
        lambda cluster_objs: cluster_objs.apply(
            lambda obj1: cluster_objs.apply(
                lambda obj2: np.linalg.norm(obj1[:-1] - obj2[:-1]), axis=1).max(), axis=1)).max()
    return min_dist_between_clusters / max_cluster_diameter


def DBI(df):
    # расстояние от объектов кластера до их центроидов
    def calc_S(cluster_objs, cluster_center):
        return cluster_objs.apply(
            lambda obj: np.linalg.norm(cluster_center - obj), axis=1).mean()

    # итог
    def score(c1, c2):
        cluster_objs1 = df[df.cluster == c1].drop('cluster', axis=1)
        cluster_center1 = cluster_objs1.mean()
        cluster_objs2 = df[df.cluster == c2].drop('cluster', axis=1)
        cluster_center2 = cluster_objs2.mean()
        return (calc_S(cluster_objs1, cluster_center1) + calc_S(cluster_objs2, cluster_center2)) / np.linalg.norm(
            cluster_center1 - cluster_center2)

    clusters = df.cluster.unique()
    return pd.Series([
        pd.Series([score(c1, c2) for c2 in clusters if c1 < c2]).max() for c1 in clusters]).mean()


# https://www.dbs.ifi.lmu.de/~zimek/publications/SDM2014/DBCV.pdf
def DBCV(df):
    clusters = df.cluster.unique()
    all_objs_count = df.shape[0]

    # DSPC(Ci, Cj) is the minimum distance between any pair of points where one belongs to Ci and the other to Cj
    def DSPC(c1, c2):
        return get_min_dist_between_clusters(df, c1, c2)

    # DSC(Ci) is the longest edge in the mutual-reachability MST of Ci (MST - мин. остовное дерево)
    def DSC(c1):
        edge_longs = []
        cluster_objs = df[df.cluster == c1].drop('cluster', axis=1)
        selected_nodes = set()
        selected_nodes.add(next(cluster_objs.iterrows())[0])
        for i in range(0, len(cluster_objs) - 1):
            pot_new_nodes = []
            for node in selected_nodes:
                obj1 = cluster_objs.loc[node, :]
                df2 = cluster_objs[(cluster_objs.index != node) & (~cluster_objs.index.isin(selected_nodes))].apply(
                    lambda obj2: np.linalg.norm(obj1 - obj2), axis=1)
                pot_new_nodes.append((df2.idxmin(), df2.min()))
            new_node = min(pot_new_nodes, key=lambda x: x[1])
            selected_nodes.add(new_node[0])
            edge_longs.append(new_node[1])
        return max(edge_longs)

    # нормализованный вес
    def V(c1):
        min_dspc = pd.Series([DSPC(c1, c2) for c2 in clusters if c2 != c1]).min()
        dsc = DSC(c1)
        return (min_dspc - dsc) / max(min_dspc, dsc)

    # итог
    def score(c1):
        cluster_objs_count = df[df.cluster == c1].shape[0]
        return cluster_objs_count * V(c1) / all_objs_count

    return pd.Series([score(c1) for c1 in clusters]).sum()


def estimate_clst(clusters):
    df = to_df(clusters)
    score = CoefficientOfDetermination(df)
    print(f'1. Коэффициент детерминации = {score}')
    score = Silhouette(df)
    print(f'2. Коэффициент силуэта = {score}')
    score = DunnIndex(df)
    print(f'3. Индекс Данна (Dunn index) = {score}')
    score = DBI(df)
    print(f'4. Индекс Девиса-Болдина (DBI) = {score}')
    score = DBCV(df)
    print(f'5. Индекс валидности по плотности кластеризации (DBCV) = {score}')







def main():
    # загружаем множество объектов
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target).map(lambda label: -1.0 if label == 2 else 1.0)
    # положительный класс имеет метку 1.0

    maxmin_method = MaxminMethod(X)
    maxmin_method.start()
    clusters1 = maxmin_method.get_clusters()

    k_means_method = KMeansMethod(X, 3)
    k_means_method.start()
    clusters2 = k_means_method.get_clusters()

    # делим множество на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=5)

    classifier1 = KNearestNeighbors(5)
    classifier1.train(X_train, y_train)
    predicted1 = classifier1.predict_test_set(X_test)

    classifier2 = SVM()
    classifier2.train(X_train, y_train)
    predicted2 = classifier2.predict_test_set(X_test)

    print('KNearestNeighbors')
    estimate_clf(y_test, predicted1)
    accuracy = cross_validation(classifier1, X, y)
    print(f'5) Кросс-валидация = {accuracy}')

    print('SVM')
    estimate_clf(y_test, predicted2)
    accuracy = cross_validation(classifier2, X, y)
    print(f'5) Кросс-валидация = {accuracy}')

    print('MaxminMethod')
    estimate_clst(clusters1)

    print('KMeansMethod')
    estimate_clst(clusters2)


if __name__ == "__main__":
    main()