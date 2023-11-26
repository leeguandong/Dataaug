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


def opencv_show(img):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class PerspectiveAugment():
    def __init__(self, is_perspective=True):
        self.is_perspective = is_perspective

    def coordinateTransform(self, pts, m):
        num_pt = pts.shape[0]
        tmp_pts = np.ones((num_pt, 3))
        tmp_pts[:, :2] = pts
        trans_pts = np.dot(tmp_pts, m.T)
        trans_pts[:, 0] = trans_pts[:, 0] / trans_pts[:, 2]
        trans_pts[:, 1] = trans_pts[:, 1] / trans_pts[:, 2]
        trans_pts = trans_pts[:, :2].reshape(-1, 2).tolist()

        return trans_pts

    def rad(self, x):
        return x * np.pi / 180

    def random_perspect(self, img, x_ratio=0.8, y_ratio=0.8, z_ratio=0.8, angle_x=20, angle_y=20, angle_z=20, fov=42):
        # img = cv2.imread(img_path)
        # img = cv2.imdecode(np.fromfile(str(img_path)), cv2.IMREAD_UNCHANGED)
        # img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        # 扩展图像，保证内容不超出可视范围
        img = cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_CONSTANT, 0)
        w, h = img.shape[0:2]

        anglex = 0
        angley = 0
        anglez = 0  # 旋转

        if random.random() < x_ratio:
            anglex = random.randint(-angle_x, angle_x)
        if random.random() < y_ratio:
            angley = random.randint(-angle_y, angle_y)
        if random.random() < z_ratio:
            anglez = random.randint(-angle_z, angle_z)

        z = np.sqrt(w ** 2 + h ** 2) / 2 / np.tan(self.rad(fov / 2))

        # 齐次变换矩阵
        rx = np.array([[1, 0, 0, 0],
                       [0, np.cos(self.rad(anglex)), -np.sin(self.rad(anglex)), 0],
                       [0, -np.sin(self.rad(anglex)), np.cos(self.rad(anglex)), 0, ],
                       [0, 0, 0, 1]], np.float32)

        ry = np.array([[np.cos(self.rad(angley)), 0, np.sin(self.rad(angley)), 0],
                       [0, 1, 0, 0],
                       [-np.sin(self.rad(angley)), 0, np.cos(self.rad(angley)), 0, ],
                       [0, 0, 0, 1]], np.float32)

        rz = np.array([[np.cos(self.rad(anglez)), np.sin(self.rad(anglez)), 0, 0],
                       [-np.sin(self.rad(anglez)), np.cos(self.rad(anglez)), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]], np.float32)

        r = rx.dot(ry).dot(rz)

        # 四对点的生成
        pcenter = np.array([h / 2, w / 2, 0, 0], np.float32)

        p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
        p2 = np.array([w, 0, 0, 0], np.float32) - pcenter
        p3 = np.array([0, h, 0, 0], np.float32) - pcenter
        p4 = np.array([w, h, 0, 0], np.float32) - pcenter

        dst1 = r.dot(p1)
        dst2 = r.dot(p2)
        dst3 = r.dot(p3)
        dst4 = r.dot(p4)

        list_dst = [dst1, dst2, dst3, dst4]

        org = np.array([[0, 0],
                        [w, 0],
                        [0, h],
                        [w, h]], np.float32)

        dst = np.zeros((4, 2), np.float32)

        # 投影至成像平面
        for i in range(4):
            dst[i, 0] = list_dst[i][0] * z / (z - list_dst[i][2]) + pcenter[0]
            dst[i, 1] = list_dst[i][1] * z / (z - list_dst[i][2]) + pcenter[1]

        warpR = cv2.getPerspectiveTransform(org, dst)
        trans_img = cv2.warpPerspective(img, warpR, (h, w))

        return trans_img, warpR

    def _perspective_pic_bboxes(self, img, json_info):
        trans_img, warpR = self.random_perspect(img)
        # opencv_show(trans_img)
        shapes = json_info['shapes']
        for shape in shapes:
            vertexes = np.array(shape["points"], dtype=np.float32).reshape(-1, 2)
            for point in vertexes:
                point[0] += 100
                point[1] += 100
            trans_vertexes = self.coordinateTransform(vertexes, warpR)
            shape["points"] = trans_vertexes

        # 校验坐标超出透视变换图
        h, w, _ = trans_img.shape
        x_min = 0
        x_max = w
        y_min = 0
        y_max = h
        for shape in shapes:
            points = np.array(shape['points'])
            x_min = min(x_min, points[:, 0].min())
            y_min = min(y_min, points[:, 1].min())
            x_max = max(x_max, points[:, 0].max())
            y_max = max(y_max, points[:, 1].max())

        if x_min < 0 or y_min < 0:
            return trans_img, None
        elif x_max > w or y_max > h:
            return trans_img, None
        return trans_img, json_info

    def dataAugment(self, img, dic_info):
        if self.is_perspective:
            out_img, dic_info = self._perspective_pic_bboxes(img, dic_info)
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
    parser.add_argument('--source_img_json_path', type=str, default='E:/qiantupsd/viewable_area/mcbv1')
    parser.add_argument('--save_img_json_path', type=str, default='E:/qiantupsd/viewable_area/mcbv1_transform')
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
                    try:
                        auged_img, json_info = dataAug.dataAugment(deepcopy(img), deepcopy(json_dic))
                    except:
                        cnt += 1
                        continue
                    if json_info == None:
                        continue
                    img_name = '{}_trans_{}{}'.format(_file_prefix, cnt + 1, _file_suffix)  # 图片保存的信息
                    img_save_path = os.path.join(save_img_json_path, img_name)
                    toolhelper.save_img(img_save_path, auged_img)  # 保存增强图片

                    json_info['imagePath'] = img_name
                    base64_data = toolhelper.img2str(img_save_path)
                    json_info['imageData'] = base64_data
                    toolhelper.save_json('{}_trans_{}.json'.format(_file_prefix, cnt + 1), save_img_json_path, json_info)  # 保存json文件
                    print(img_name)
                    cnt += 1  # 继续增强下一张
