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
from copy_paste import copy_paste
from PIL import Image


class CopyPasteAugment():
    def __init__(self, is_copy=True):
        self.is_copy = is_copy

    def _copy_paste(self, img, background_type, alignment, json_info):
        img, coord = copy_paste(img, background_type, alignment)
        shapes = json_info['shapes']
        for shape in shapes:
            for p in shape['points']:
                p[0] += coord[0]
                p[1] += coord[1]
        return img, json_info

    def dataAugment(self, img, dic_info):
        background_type = 3
        num = random.randint(0, 2)

        if self.is_copy:
            out_img, dic_info = self._copy_paste(img, background_type, num, dic_info)  # target_color, base_color  #1FE29D, 6646DD
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
        # cv2.imwrite(save_path, img)
        cv2.imencode('.jpg', img)[1].tofile(save_path)

    # 保持json结果
    def save_json(self, file_name, save_folder, dic_info):
        with open(os.path.join(save_folder, file_name), 'w') as f:
            json.dump(dic_info, f, indent=2)


if __name__ == '__main__':
    need_aug_num = 10  # 每张图片需要增强的次数
    toolhelper = ToolHelper()  # 工具
    is_endwidth_dot = True  # 文件是否以.jpg或者png结尾
    dataAug = CopyPasteAugment()  # 数据增强工具类

    # 获取相关参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_img_json_path', type=str, default='E:/qiantupsd/viewable_area/png_test')
    parser.add_argument('--save_img_json_path', type=str, default='E:/qiantupsd/viewable_area/png_copy')
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
                # img = cv2.imdecode(np.fromfile(pic_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                img = Image.open(pic_path)

                while cnt < need_aug_num:  # 继续增强
                    try:
                        auged_img, json_info = dataAug.dataAugment(deepcopy(img), deepcopy(json_dic))
                    except:
                        cnt += 1
                        continue
                    img_name = '{}_copy_{}{}'.format(_file_prefix, cnt + 1, _file_suffix)  # 图片保存的信息
                    img_save_path = os.path.join(save_img_json_path, img_name)
                    toolhelper.save_img(img_save_path, auged_img)  # 保存增强图片

                    json_info['imagePath'] = img_name
                    base64_data = toolhelper.img2str(img_save_path)
                    json_info['imageData'] = base64_data
                    toolhelper.save_json('{}_copy_{}.json'.format(_file_prefix, cnt + 1), save_img_json_path, json_info)  # 保存json文件
                    print(img_name)
                    cnt += 1  # 继续增强下一张
