import cv2
import re
import numpy as np
from glob import glob
from lab8 import process_image
import pickle


N = 0
mask = None


def find_left_down_pixel(img):
    crop_left = 0
    crop_down = 0
    for i in range(img.shape[1]):
        if img[:, i].sum() > 0:
            crop_left = i
            break
    for i in reversed(range(img.shape[0])):
        if img[i, :crop_left + 1].sum() > 0:
            crop_down = i
            break
    return crop_down, crop_left


def sign(img):
    DIRECTIONS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    sign = '('
    img_map = img > 0
    cur_pos = find_left_down_pixel(img)
    pos_lst = []
    pos_lst_set = set()
    while cur_pos:
        img_map[cur_pos] = False
        next_posistions = []
        is_first = True
        for dir_idx, direction in enumerate(DIRECTIONS):
            next_pos = (cur_pos[0] + direction[0], cur_pos[1] + direction[1])
            if not (0 <= next_pos[0] < img_map.shape[0] and 0 <= next_pos[1] < img_map.shape[1]):
                continue
            if not img_map[next_pos]:
                continue
            if not is_first and next_pos in pos_lst_set:
                continue
            next_posistions.append((next_pos, dir_idx + 1))
            pos_lst_set.add(next_pos)
            is_first = False
        if not next_posistions:
            cur_pos = None
            for next_pos, next_dir_idx in reversed(pos_lst):
                if img_map[next_pos]:
                    cur_pos = next_pos
                    sign += ')(0' + str(next_dir_idx)
                    break
        else:
            cur_pos, next_dir_idx = next_posistions.pop(0)
            pos_lst.extend(next_posistions)
            sign += str(next_dir_idx)

    return sign + ')'


def normalize(sign):
    sign = sign.replace('0', '').replace('(0)', '')
    prev_sign = ''
    while sign != prev_sign:
        prev_sign = sign
        for i in range(1, 10):
            sign = re.sub(f'{i}\d?{i}+', f'{i}', sign)
    return sign


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
        signs = []
        for filename in filenames:
            img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            #img = cv2.Canny(img, 100, 200)
            norm_sign = normalize(sign(img))
            if len(norm_sign) > 5:
                signs.append(norm_sign)
        symbol_map[symbol] = signs
    return symbol_map


def predict(symbol_map, img_sign):
    for symbol, symbol_signs in symbol_map.items():
        if img_sign in symbol_signs:
            return symbol
    return None


def save_model(symbol_map, file):
    pickle.dump(symbol_map, open(file, 'wb'), pickle.HIGHEST_PROTOCOL)


def load_model(file):
    return pickle.load(open(file, "rb"))


if __name__ == "__main__":
    N = 10
    # a) обучение
    new_symbol_map = create_symbol_map(f'processed_{N}\\')
    save_model(new_symbol_map, 'symbol_map1.model')

    # b) распознавание
    symbol_map = load_model('symbol_map1.model')

    test = ['0', '1', '1_2', '2', '5', 'A', 'C', 'R', 'N', 'N_2', 'U', 'S', 'D', 'D_2']
    preds = []
    for symbol in test:
        img = process_image(cv2.imread(f'predict\\{symbol}.jpg', cv2.IMREAD_UNCHANGED), N)
        #img = cv2.Canny(img, 100, 200)
        img_sign = normalize(sign(img))
        pred_symbol = predict(symbol_map, img_sign)
        preds.append((symbol, pred_symbol))
    print(preds)