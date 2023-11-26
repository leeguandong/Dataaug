# -*- coding: utf-8 -*-
import cv2
import numpy as np
from skimage import color
from matplotlib.colors import hex2color, rgb2hex, rgb_to_hsv
import math
from PIL import Image
import copy


# import operator
# import time

def roundInt(float_):
    if float_ - int(float_) >= 0.5:
        return int(float_) + 1
    else:
        return int(float_)


def color_transfer(img, target_color, base_color=None):
    # @param img: ndarray in cv2 format.
    # @param target_color: hex code of rgb value you wanna go.
    # @param base_color: hex code of rgb value of your original template.

    # convert target color from RGB to LCH
    target_lch = color.lab2lch(color.rgb2lab(np.reshape(hex2color(target_color), (1, 1, 3)))).flatten()

    # convert template color from RGB to LCH
    # if not given, extract the sub-elements' own color.
    override_flag = 0
    has_w_b_flag = 0
    appointed_layer_flag = 0
    top_color = 0
    theme_lch_temp = None

    img_sample = cv2.resize(img, (40, 40), interpolation=cv2.INTER_AREA)

    theme_rgbs = get_theme_color(Image.fromarray(cv2.cvtColor(img_sample, cv2.COLOR_BGRA2RGBA)), 3)
    for theme_rgb in theme_rgbs:
        theme_rgb = np.reshape(theme_rgb, (1, 1, 3)) / 255.
        theme_lch = color.lab2lch(fast_rgb2lab(theme_rgb)).flatten()
        if top_color == 0:
            top_color = 1
            theme_lch_temp = theme_lch
        if theme_lch[0] < 10 or theme_lch[0] > 90 or theme_lch[1] < 11:  # it is blk or wht
            has_w_b_flag = 1
    theme_lch = copy.deepcopy(theme_lch_temp)

    if base_color:  # if base color is appointed by usr, c_trans tends to be strictly match with target color
        appointed_layer_flag = 1  # this layer tends to be the leader layer
        override_flag = 1
        theme_rgb = hex2color(base_color)
        theme_lch_temp = color.lab2lch(fast_rgb2lab(np.reshape(theme_rgb, (1, 1, 3)))).flatten()
        for pair in zip(theme_lch, theme_lch_temp):
            if pair[0] != pair[1]:  # this layer actually not the leader layer
                appointed_layer_flag = 0
        theme_lch = theme_lch_temp

    l_gap = target_lch[0] - theme_lch[0]
    c_gap = target_lch[1] - theme_lch[1]
    h_gap = target_lch[2] - theme_lch[2]
    if appointed_layer_flag == 0 and has_w_b_flag == 1:  # 不是指定迁移图层 并且 主色含有白或黑， 进入旧逻辑
        override_flag = 0

    # # 亮度为黑白（x<10 或x>90）时，权重小于0.1，保持黑白
    # if theme_lch[0] < 10 or theme_lch[0] > 90:
    #     override_flag = 0
    # # 饱和度为黑白（x<11）时，权重小于0.1，保持黑白
    # if theme_lch[1] < 11:
    #     override_flag = 0

    new_img = item_transfer(img, l_gap, c_gap, h_gap, override_flag, theme_lch, target_lch, hex2color(target_color), theme_rgb)

    return new_img


def text_color_transfer(text_color, target_color, base_color=None):
    target_lch = color.lab2lch(fast_rgb2lab(np.reshape(
        hex2color(target_color), (1, 1, 3)))).flatten()
    text_lch = color.lab2lch(fast_rgb2lab(np.reshape(
        hex2color(text_color), (1, 1, 3)))).flatten()

    if not base_color:
        theme_lch = text_lch
    else:
        theme_lch = color.lab2lch(fast_rgb2lab(
            np.reshape(hex2color(base_color), (1, 1, 3)))).flatten()

    def fl(x):
        return -np.tanh(abs((x - 50) / 27)) + \
               1  # 亮度为黑白（x<10 或x>90）时，权重小于0.1，保持黑白

    def fs(x):
        return (np.tanh((x - 55) / 41) + 1) / \
               2  # 饱和度为黑白（x<11）时，权重小于0.1，保持黑白

    new_text_lch = np.copy(text_lch)
    gap = target_lch - theme_lch

    new_text_lch[0] += gap[0] * fl(new_text_lch[0])
    new_text_lch[1] += gap[1] * fs(new_text_lch[1])
    new_text_lch[2] = (gap[2] + new_text_lch[2]) % (2 * math.pi)

    new_text_rgb = color.lab2rgb(color.lch2lab(np.reshape(new_text_lch, (1, 1, 3)))).flatten()

    new_text_color = rgb2hex(new_text_rgb)

    return new_text_color


def item_transfer(img, l_gap, c_gap, h_gap, override_flag, theme_lch=None, target_lch=None, target_rgb=None, base_rgb=None):
    alpha = None
    if img.shape[-1] == 4:
        alpha = np.reshape(img[:, :, -1], (img.shape[0], img.shape[1], -1))

    # 20% time
    lch_img = color.lab2lch(fast_rgb2lab(img[:, :, 2::-1] / 255.))  # BGR->RGB->LAB->LCH

    # ------------------------------------------------------------------------------------------- #ff8b00 #FA8900
    def fl(x):
        if override_flag == 1:
            return 1
        else:
            return -np.tanh(abs((x - 50) / 27)) + 1

    def fs(x):
        if override_flag == 1:
            return 1
        else:
            return (np.tanh((x - 55) / 41) + 1) / 2

    # 20% time
    lch_img[:, :, 0] += l_gap * fl(copy.deepcopy(lch_img[:, :, 0]))

    lch_img[:, :, 1] += c_gap * fs(copy.deepcopy(lch_img[:, :, 1]))

    lch_img[:, :, 2] = (lch_img[:, :, 2] + h_gap) % (2 * math.pi)

    # -------------------------------------------------------------------------------------------


    # 50% time
    bgr_img = color.lab2rgb(color.lch2lab(lch_img))[:, :, 2::-1] * 255

    if alpha is not None:
        out_img = np.concatenate((bgr_img, alpha), -1)
    else:
        out_img = bgr_img
    # print(f'!!!!!!!!!!!!! func target_transfer takes {time.time() - tic_all} s')
    return out_img


def fast_rgb2lab(img):
    # 读进来的像素是经color.处理，值域在[0,1]。
    # 返回的像素将作为skimage范围为[0-100], [-128,127], [-128,127]
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise Exception("Image's shape wrong!")

    img = np.uint8(img * 255)

    # cv2出的结果是np.uin8,做类型转换
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float64)

    # 因为cv2转换完结果在0-255，归一化到目标范围
    img[:, :, 0] /= 2.55
    img[:, :, 1] -= 128.
    img[:, :, 2] -= 128.

    return img


# theme color extraction works for image with alpha channel by drop all transparent pixcel.
def get_theme_color(img, top=1, quantize_size=10, quantize_algo=2):
    if img.mode == 'RGB':
        img_quantized = img.quantize(quantize_size, quantize_algo)
    elif img.mode == 'RGBA':
        img_arr = np.asarray(img)
        alpha = img_arr[:, :, -1]
        mask = np.where(alpha == 0, False, True)
        img = Image.fromarray(img_arr[mask].reshape(1, -1, 4))
        img_quantized = img.quantize(quantize_size, quantize_algo)
    else:
        raise ValueError('neither RGB nor RGBA.')

    palette = img_quantized.getpalette()
    rgbs = img_quantized.getcolors()

    indice = sorted(rgbs, key=lambda x: -x[0])
    colors = []

    for i in range(top):
        try:
            colors.append(palette[indice[i][1] * 3:(indice[i][1] + 1) * 3])
        except:
            "do nothing"

    return colors


def fast_lab2rgb(img):
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise Exception("Image's shape wrong!")
    img[:, :, 0] *= 2.55
    img[:, :, 1] += 128.
    img[:, :, 2] += 128.

    img = cv2.cvtColor(np.uint8(img), cv2.COLOR_LAB2RGB)
    return img
