import cv2
import numpy as np


def process_image(src_img, N):
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