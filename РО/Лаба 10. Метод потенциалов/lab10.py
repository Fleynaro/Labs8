import cv2
import numpy as np
from glob import glob
from lab8 import process_image
import pickle


N = 0
mask = None


def create_mask(m):
    mask = np.zeros((2*m + 1, 2*m + 1))
    center = np.array([m, m])
    mask[m, m] = 1.0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            dist = np.linalg.norm(np.array([i, j]) - center) + 1.0
            mask[i, j] = 1 / dist
    return mask


def image_to_vector(img):
    sl = lambda x, n: slice(x, n) if x >= 0 else slice(0, x)

    result = np.zeros(img.shape)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            offset_i = i - mask.shape[0] // 2
            offset_j = j - mask.shape[1] // 2
            result[sl(offset_i, img.shape[0]), sl(offset_j, img.shape[1])] +=\
                img[sl(-offset_i, img.shape[0]), sl(-offset_j, img.shape[1])] * mask[i, j]
    return result.reshape(-1)


def create_symbol_map(dir_path):
    symbol_files = {}
    for symbol_dir in glob(f"{dir_path}*\\"):
        symbol = symbol_dir.split('\\')[-2]
        for img_file in glob(f"{symbol_dir}*"):
            if symbol not in symbol_files:
                symbol_files[symbol] = []
            symbol_files[symbol].append(img_file)

    symbol_map = {}
    for symbol, filenames in symbol_files.items():
        vectors = []
        for filename in filenames:
            img = cv2.imread(filename, cv2.IMREAD_UNCHANGED).astype(float) / 255.0
            vectors.append(image_to_vector(img))
        symbol_map[symbol] = vectors
    return symbol_map


def predict(symbol_map, img_vector, alpha=1.0):
    scores = []
    for symbol, symbol_vectors in symbol_map.items():
        squared_dist_vec = np.array(list(map(lambda x: np.linalg.norm(x - img_vector), symbol_vectors)))
        phi_avg = (1. / (1 + alpha * np.power(squared_dist_vec, 2.0))).mean()
        scores.append((symbol, phi_avg))
    return max(scores, key=lambda x: x[1])[0]


def predict2(symbol_map, img_vector, alpha=1.0):
    scores = []
    for symbol, symbol_vectors in symbol_map.items():
        squared_dist_vec = np.array(list(map(lambda x: np.linalg.norm(x - img_vector), symbol_vectors)))
        scores.append((symbol, squared_dist_vec.min()))
    return min(scores, key=lambda x: x[1])[0]


def save_model(symbol_map, file):
    pickle.dump(symbol_map, open(file, 'wb'), pickle.HIGHEST_PROTOCOL)


def load_model(file):
    return pickle.load(open(file, "rb"))


if __name__ == "__main__":
    N = 10
    mask = np.array([
        [1/12, 1/6, 1/12],
        [1/6, 1, 1/6],
        [1/12, 1/6, 1/12],
    ])
    mask = create_mask(2)

    # a) обучение
    new_symbol_map = create_symbol_map(f'processed_{N}\\')
    save_model(new_symbol_map, 'symbol_map2.model')

    # b) распознавание
    symbol_map = load_model('symbol_map2.model')

    test = ['0', '1', '1_2', '2', '5', 'A', 'C', 'R', 'N', 'N_2', 'U', 'S', 'D', 'D_2']
    preds = []
    for symbol in test:
        img_vector = image_to_vector(
            process_image(cv2.imread(f'predict\\{symbol}.jpg', cv2.IMREAD_UNCHANGED), N).astype(float) / 255.0)
        pred_symbol = predict(symbol_map, img_vector)
        preds.append((symbol, pred_symbol))
    print(preds)