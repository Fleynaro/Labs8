import multiprocessing
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

N = 50
processors_count = 1
process_idx = 0


def get_all_images(dir):
    symbol_files = {}
    for guid_dir in glob(f"{dir}*\\"):
        for symbol_dir in glob(f"{guid_dir}*\\"):
            symbol = symbol_dir.split('\\')[-2]
            for img_file in glob(f"{symbol_dir}*"):
                if symbol not in symbol_files:
                    symbol_files[symbol] = []
                symbol_files[symbol].append(img_file)
    return symbol_files


def crop_image(img):
    crop_left = 0
    crop_right = 0
    crop_up = 0
    crop_down = 0
    for i in range(img.shape[1]):
        if img[:, i].sum() != 0:
            crop_left = i
            break
    for i in reversed(range(img.shape[1])):
        if img[:, i].sum() != 0:
            crop_right = i + 1
            break
    for i in range(img.shape[0]):
        if img[i, :].sum() != 0:
            crop_up = i
            break
    for i in reversed(range(img.shape[0])):
        if img[i, :].sum() != 0:
            crop_down = i + 1
            break
    return img[crop_up:crop_down, crop_left:crop_right]


def process_image(src_img):
    # монохромное изображение
    im_bw = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    if (im_bw == 0).all():
        # исключительный случай
        im_bw = np.zeros((src_img.shape[0], src_img.shape[1]), np.uint8)
        im_bw[:, :] = (src_img.max(axis=2) > 0) * 255
    else:
        _, im_bw = cv2.threshold(im_bw, im_bw.mean(), 255, cv2.THRESH_BINARY)
        if im_bw[0, 0] == 255:
            im_bw = cv2.bitwise_not(im_bw)

    # вырезаем символ
    cropped_img = crop_image(im_bw)

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


def work(i):
    global process_idx
    process_idx = i
    symbol_files = get_all_images('train\\')
    processed_images = {}
    for symbol, filenames in symbol_files.items():
        for filename in filenames:
            print(filename)
            img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            if img is None:
                print('warning: image not opened.')
                continue
            result_img = process_image(img)
            if symbol not in processed_images:
                processed_images[symbol] = []
            processed_images[symbol].append(result_img)

    for symbol, images in processed_images.items():
        symbol_dir = f'processed\\{symbol}'
        if not os.path.exists(symbol_dir):
            os.makedirs(symbol_dir)
        for i, img in enumerate(images):
            img_id = str(10000 + i)[1:]
            cv2.imwrite(f'{symbol_dir}\\{img_id}.png', img)


def main():
    data = list(range(processors_count))
    pool = multiprocessing.Pool(processors_count)
    mapped = pool.map(work, data)
    pool.close()
    pool.join()
    return mapped


if __name__ == "__main__":
    main()