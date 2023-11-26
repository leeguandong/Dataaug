import time
import random
import cv2
import os
import numpy as np
import base64
import json
import re
from copy import deepcopy
import argparse


class PerspectiveAugment():
    def __init__(self, is_perspective=True):
        self.is_perspective = is_perspective

    # 透视变换
    def _shift_pic_bboxes(self, img, json_info):
        # ---------------------- 平移图像 ----------------------
        h, w, _ = img.shape
        x_min = w
        x_max = 0
        y_min = h
        y_max = 0

        shapes = json_info['shapes']
        for shape in shapes:
            points = np.array(shape['points'])
            x_min = min(x_min, points[:, 0].min())
            y_min = min(y_min, points[:, 1].min())
            x_max = max(x_max, points[:, 0].max())
            y_max = max(y_max, points[:, 1].max())

        d_to_left = x_min  # 包含所有目标框的最大左移动距离
        d_to_right = w - x_max  # 包含所有目标框的最大右移动距离
        d_to_top = y_min  # 包含所有目标框的最大上移动距离
        d_to_bottom = h - y_max  # 包含所有目标框的最大下移动距离

        w_to_left = w - 20
        w_to_right = w
        h_to_top = h - 20
        h_to_bottom = h

        x = random.uniform(-(d_to_left - 1) / 3, (d_to_right - 1) / 3)
        y = random.uniform(-(d_to_top - 1) / 3, (d_to_bottom - 1) / 3)
        w = random.uniform(-(w_to_left - 1) / 3, (w_to_right - 1) / 3)
        h = random.uniform(-(h_to_top - 1) / 3, (h_to_bottom - 1) / 3)

        # 仿射变换
        # M = np.float32([[1, 0, x], [0, 1, y]])  # x为向左或右移动的像素值,正为向右负为向左; y为向上或者向下移动的像素值,正为向下负为向上
        # shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        # 透视变换
        post1 = np.float32([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
        post2 = np.float32([[x_min + x - w, y_min + y - h], [x_max + x - w, y_min + y - h], [x_max + x - w, y_max + y - h],
                            [x_min + x - w, y_max + y - h]])
        M = cv2.getPerspectiveTransform(post1, post2)
        shift_img = cv2.warpPerspective(img, M, (img.shape[1], img.shape[2]))

        # ---------------------- 平移boundingbox ----------------------
        for shape in shapes:
            for p in shape['points']:
                p[0] += (x - w)
                p[1] += (y - h)
        return shift_img, json_info

    def dataAugment(self, img, dic_info):
        if self.is_perspective:
            out_img, dic_info = self._shift_pic_bboxes(img, dic_info)  # target_color, base_color  #1FE29D, 6646DD
        return out_img, dic_info


class ToolHelper():
    # 从json文件中提取原始标定的信息
    def parse_json(self, path):
        with open(path)as f:
            json_data = json.load(f)
        return json_data

    # 对图片进行字符编码
    def img2str(self, img_name):
        with open(img_name, "rb")as f:
            base64_data = str(base64.b64encode(f.read()))
        match_pattern = re.compile(r'b\'(.*)\'')
        base64_data = match_pattern.match(base64_data).group(1)
        return base64_data

    # 保存图片结果
    def save_img(self, save_path, img):
        cv2.imwrite(save_path, img)

    # 保持json结果
    def save_json(self, file_name, save_folder, dic_info):
        with open(os.path.join(save_folder, file_name), 'w') as f:
            json.dump(dic_info, f, indent=2)


if __name__ == '__main__':
    need_aug_num = 10  # 每张图片需要增强的次数
    toolhelper = ToolHelper()  # 工具
    is_endwidth_dot = True  # 文件是否以.jpg或者png结尾
    dataAug = PerspectiveAugment()  # 数据增强工具类

    # 获取相关参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_img_json_path', type=str, default='E:/qiantupsd/viewable_area/png')
    parser.add_argument('--save_img_json_path', type=str, default='E:/qiantupsd/viewable_area/png_transform')
    args = parser.parse_args()
    source_img_json_path = args.source_img_json_path  # 图片和json文件原始位置
    save_img_json_path = args.save_img_json_path  # 图片增强结果保存文件

    # 如果保存文件夹不存在就创建
    if not os.path.exists(save_img_json_path):
        os.mkdir(save_img_json_path)

    for parent, _, files in os.walk(source_img_json_path):
        files.sort()  # 排序一下
        for file in files:
            if file.endswith('jpg') or file.endswith('png'):
                cnt = 0
                pic_path = os.path.join(parent, file)
                json_path = os.path.join(parent, file[:-4] + '.json')
                json_dic = toolhelper.parse_json(json_path)
                # 如果图片是有后缀的
                if is_endwidth_dot:
                    # 找到文件的最后名字
                    dot_index = file.rfind('.')
                    _file_prefix = file[:dot_index]  # 文件名的前缀
                    _file_suffix = file[dot_index:]  # 文件名的后缀
                # img = cv2.imread(pic_path)
                img = cv2.imdecode(np.fromfile(pic_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                # img = Image.open(pic_path)

                while cnt < need_aug_num:  # 继续增强
                    auged_img, json_info = dataAug.dataAugment(deepcopy(img), deepcopy(json_dic))
                    img_name = '{}_{}{}'.format(_file_prefix, cnt + 1, _file_suffix)  # 图片保存的信息
                    img_save_path = os.path.join(save_img_json_path, img_name)
                    toolhelper.save_img(img_save_path, auged_img)  # 保存增强图片

                    json_info['imagePath'] = img_name
                    base64_data = toolhelper.img2str(img_save_path)
                    json_info['imageData'] = base64_data
                    toolhelper.save_json('{}_{}.json'.format(_file_prefix, cnt + 1), save_img_json_path, json_info)  # 保存json文件
                    print(img_name)
                    cnt += 1  # 继续增强下一张
