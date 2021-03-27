import multiprocessing
import os
import cv2
import re
import numpy as np
import matplotlib.pyplot as plt
from glob import glob


N = 0


def process_image(src_img):
    # монохромное изображение
    im_bw = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    if (im_bw == 0).all():
        # исключительный случай
        im_bw = np.zeros((src_img.shape[0], src_img.shape[1]), np.uint8)
        im_bw[:, :] = (src_img.max(axis=2) > 0) * 255
    else:
        _, im_bw = cv2.threshold(im_bw, im_bw.mean(), 255, cv2.THRESH_BINARY)
        if im_bw[-1, -1] == 255:
            im_bw = cv2.bitwise_not(im_bw)

    # вырезаем символ
    def crop_image(img, threshold):
        crop_left = 0
        crop_right = 0
        crop_up = 0
        crop_down = 0
        for i in range(img.shape[1]):
            if img[:, i].sum() > threshold:
                crop_left = i
                break
        for i in reversed(range(img.shape[1])):
            if img[:, i].sum() > threshold:
                crop_right = i + 1
                break
        for i in range(img.shape[0]):
            if img[i, :].sum() > threshold:
                crop_up = i
                break
        for i in reversed(range(img.shape[0])):
            if img[i, :].sum() > threshold:
                crop_down = i + 1
                break
        return img[crop_up:crop_down, crop_left:crop_right]
    cropped_img = crop_image(im_bw, 1000)

    # ресайзим до NxN
    new_size = (N * cropped_img.shape[1] // max(cropped_img.shape),
                N * cropped_img.shape[0] // max(cropped_img.shape))
    resized_img = cv2.resize(cropped_img, new_size, cv2.INTER_NEAREST)
    resized_img = (resized_img > 200) * 255

    # создаем пустое изображение размера NxN и вставляем символ по центру
    result_img = np.zeros((N, N), np.uint8)
    offset_x = (N - resized_img.shape[1]) // 2
    offset_y = (N - resized_img.shape[0]) // 2
    result_img[offset_y:offset_y + resized_img.shape[0], offset_x:offset_x + resized_img.shape[1]] = resized_img
    return result_img


def image_to_vector2(img):
    mask = np.array([
        [1. / 6, 1. / 2, 1. / 6],
        [1. / 2, 1., 1. / 2],
        [1. / 6, 1. / 2, 1. / 6]
    ])
    mask = np.array([
        [0.5, 0.75, 0.5],
        [0.75, 1.0, 0.75],
        [0.5, 0.75, 0.5],
    ])

    result = img.copy()
    result[:, 1:] += img[:, :-1] * mask[1, 2]
    result[:, :-1] += img[:, 1:] * mask[1, 0]
    result[1:, :] += img[:-1, :] * mask[2, 1]
    result[:-1, :] += img[1:, :] * mask[0, 1]
    result[:-1, :-1] += img[1:, 1:] * mask[0, 0]
    result[:-1, 1:] += img[1:, :-1] * mask[0, 2]
    result[1:, :-1] += img[:-1, 1:] * mask[2, 0]
    result[1:, 1:] += img[:-1, :-1] * mask[2, 2]
    return result.reshape(-1)


def image_to_vector(img):
    mask = np.array([
        [0.1, 0.3, 0.5, 0.3, 0.1],
        [0.3, 0.5, 0.8, 0.5, 0.3],
        [0.5, 0.8, 1.0, 0.8, 0.5],
        [0.3, 0.5, 0.8, 0.5, 0.3],
        [0.1, 0.3, 0.5, 0.3, 0.1],
    ])
    sl = lambda x, n: slice(x, n) if x >= 0 else slice(0, x)

    result = np.zeros(img.shape)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            offset_i = i - mask.shape[0] // 2
            offset_j = j - mask.shape[1] // 2
            result[sl(offset_i, img.shape[0]), sl(offset_j, img.shape[1])] +=\
                img[sl(-offset_i, img.shape[0]), sl(-offset_j, img.shape[1])] * mask[i, j]
    return result.reshape(-1)


def load_symbol_map(dir_path):
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


if __name__ == "__main__":
    N = 15
    symbol_map = load_symbol_map(f'processed_{N}\\')
    symbol = predict(symbol_map, image_to_vector(cv2.imread(f'processed_{N}\\5\\0003.png', cv2.IMREAD_UNCHANGED).astype(float) / 255.0))

    symbol2 = predict(symbol_map, image_to_vector(
        process_image(cv2.imread('predict\\5.jpg', cv2.IMREAD_UNCHANGED)).astype(float) / 255.0))

    pass