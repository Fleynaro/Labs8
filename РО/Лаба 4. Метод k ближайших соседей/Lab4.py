import sys
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# кластеризатор
class KNearestNeighbors:
    def __init__(self, X_train, y_train, k):
        self.X_train = X_train
        self.y_train = y_train
        self.k = k
        self.metric_func = lambda x, y: ((x - y) ** 2).sum()

    def predict(self, sample):
        indexes = self.X_train.apply(lambda row: self.metric_func(row, sample), axis=1).to_frame().sort_values(0).head(self.k).index
        return self.y_train[indexes].value_counts().idxmax()

    def predict_test_set(self, X_test):
        return X_test.apply(lambda row: self.predict(row), axis=1)


def main():
    # загружаем множество объектов
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)

    # делим множество на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # кластеризация
    classifier = KNearestNeighbors(X, y, 5)
    predicted = classifier.predict_test_set(X_test)
    accuracy = (y_test == predicted).mean()
    print(f'accuracy = {accuracy}')

if __name__ == "__main__":
    main()