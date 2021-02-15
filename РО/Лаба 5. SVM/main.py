import sys
import pandas as pd
import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def add_bias_feature(a):
    a_extended = np.zeros((a.shape[0], a.shape[1] + 1))
    a_extended[:, :-1] = a
    a_extended[:, -1] = 1.
    return a_extended


# классификатор
class SVM:
    def __init__(self):
        self.w = None

    def train(self, X_train, y_train, X_test, y_test, epoches_count=10, batch_size=5, alpha=0.1):
        # создаем вектор весов, расширяя пространство на 1
        self.w = torch.ones([X_train.shape[1] + 1, 1], requires_grad=True)

        # стохастический градиентный спуск
        optimizer = torch.optim.SGD([self.w], lr=1.0e-3)
        for epoch in range(epoches_count):
            # перемешиваем индексы
            order = np.random.permutation(len(X_train))
            for start_index in range(0, len(X_train), batch_size):
                optimizer.zero_grad()
                # формируем батч
                batch_indexes = order[start_index:start_index + batch_size]
                X_batch = torch.tensor(add_bias_feature(X_train.iloc[batch_indexes].values).astype(np.float32))
                y_batch = torch.tensor(y_train.iloc[batch_indexes].values.astype(np.float32)).unsqueeze(1)

                # построение графа вычислений для функции потерь SVM
                zeros = torch.zeros([batch_size, 1])
                L = self.w.t() @ self.w / 2.
                loss_val = torch.max(zeros, 1. - y_batch * X_batch @ self.w) + alpha / len(X_train) * L
                loss_val = loss_val.sum()

                # нахождение градиента
                loss_val.backward()
                optimizer.step()
            predicted = self.predict_test_set(X_test)
            accuracy = (y_test == predicted).mean()
            print(f'accuracy = {accuracy}')

    def predict(self, sample):
        x = sample.values
        x = np.append(x, 1.)
        label = self.w.t() @ torch.tensor(x.astype(np.float32))
        return np.sign(label.item())

    def predict_test_set(self, X_test):
        return X_test.apply(lambda row: self.predict(row), axis=1)


def main():
    # загружаем множество объектов
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target).map(lambda label: -1.0 if label == 0 else 1.0)

    # делим множество на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

    # кластеризация
    classifier = SVM()
    classifier.train(X_train, y_train, X_test, y_test)
    predicted = classifier.predict_test_set(X_test)
    accuracy = (y_test == predicted).mean()
    print(f'accuracy = {accuracy}')

if __name__ == "__main__":
    main()