import cv2
import numpy as np
from glob import glob
from lab8 import process_image
import pickle
import torch
import random


N = 0


class LeNet5(torch.nn.Module):
    def __init__(self, symbols):
        super(LeNet5, self).__init__()

        self.symbols = symbols

        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.act1 = torch.nn.Tanh()
        self.pool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = torch.nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, padding=0)
        self.act2 = torch.nn.Tanh()
        self.pool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc1 = torch.nn.Linear(5 * 5 * 16, 200)
        self.act3 = torch.nn.Tanh()

        self.fc2 = torch.nn.Linear(200, 120)
        self.act4 = torch.nn.Tanh()

        self.fc3 = torch.nn.Linear(120, len(symbols))

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        x = self.fc1(x)
        x = self.act3(x)
        x = self.fc2(x)
        x = self.act4(x)
        x = self.fc3(x)

        return x

    def predict(self, img):
        return self.symbols[self.forward(img.unsqueeze(0).unsqueeze(0)).argmax(dim=1).item()]


def load_symbols_dataset(dir_path):
    symbol_files = {}
    for symbol_dir in glob(f"{dir_path}*\\"):
        symbol = symbol_dir.split('\\')[-2]
        for img_file in glob(f"{symbol_dir}*"):
            if symbol not in symbol_files:
                symbol_files[symbol] = []
            symbol_files[symbol].append(img_file)

    X = []
    y = []
    symbols = []
    for symbol, filenames in symbol_files.items():
        symbol_idx = len(symbols)
        symbols.append(symbol)
        for filename in filenames:
            img = cv2.imread(filename, cv2.IMREAD_UNCHANGED).astype(float) / 255.0
            # = np.eye(1, len(symbol_files), k=symbol_idx, dtype=np.float32).reshape(-1)
            X.append(img)
            y.append(symbol_idx)
    return torch.Tensor(X).unsqueeze(1), torch.Tensor(y).long(), symbols


def train_model(X, y, symbols, batch_size=128, epochs=40, k=0.8):
    lenet5 = LeNet5(symbols)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lenet5.parameters(), lr=1.0e-3)

    # shuffle
    perm_idxs = np.random.permutation(len(X))
    X = X[perm_idxs]
    y = y[perm_idxs]
    # split
    train_size = round(len(X) * k)
    X_train = torch.tensor(X[:train_size])
    y_train = torch.tensor(y[:train_size])
    X_test = torch.tensor(X[train_size:])
    y_test = torch.tensor(y[train_size:])

    test_accuracy_history = []
    test_loss_history = []

    for epoch in range(epochs):
        order = np.random.permutation(len(X_train))
        for start_index in range(0, len(X_train), batch_size):
            optimizer.zero_grad()

            batch_indexes = order[start_index:start_index + batch_size]

            X_batch = X_train[batch_indexes]
            y_batch = y_train[batch_indexes]

            preds = lenet5.forward(X_batch)
            loss_value = loss(preds, y_batch)
            loss_value.backward()

            optimizer.step()

        test_preds = lenet5.forward(X_test)
        test_loss_history.append(loss(test_preds, y_test).data.cpu())

        accuracy = (test_preds.argmax(dim=1) == y_test).float().mean().data.cpu()
        test_accuracy_history.append(accuracy)
        print(accuracy)
    return lenet5


def save_model(model, file):
    pickle.dump(model, open(file, 'wb'), pickle.HIGHEST_PROTOCOL)


def load_model(file):
    return pickle.load(open(file, "rb"))


if __name__ == "__main__":
    N = 28

    # a) обучение
    if True:
        X, y, symbols = load_symbols_dataset(f'processed_{N}\\')
        model = train_model(X, y, symbols)
        save_model(model, 'lenet5.model')

    # b) распознавание
    if True:
        model = load_model('lenet5.model')

        preds = []
        for img_file in glob(f"predict\\*"):
            img = process_image(cv2.imread(img_file, cv2.IMREAD_UNCHANGED), N).astype(float) / 255.0
            img = torch.Tensor(img)
            pred_symbol = model.predict(img)
            preds.append((img_file.split('\\')[-1], pred_symbol))

        for file, pred_symbol in preds:
            print(f'{file} -> {pred_symbol}')