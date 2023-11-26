import cv2
import numpy as np
import json
import random
from pathlib import Path
from tqdm import tqdm
import shutil


def rad(x):
    return x * np.pi / 180


def coordinateTransform(pts, m):
    num_pt = pts.shape[0]
    tmp_pts = np.ones((num_pt, 3))
    tmp_pts[:, :2] = pts
    trans_pts = np.dot(tmp_pts, m.T)
    trans_pts[:, 0] = trans_pts[:, 0] / trans_pts[:, 2]
    trans_pts[:, 1] = trans_pts[:, 1] / trans_pts[:, 2]
    trans_pts = trans_pts[:, :2].reshape(-1, 2).tolist()

    return trans_pts


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def random_perspect(img_path, x_ratio=0.8, y_ratio=0.8, z_ratio=0.8, angle_x=20, angle_y=20, angle_z=20, fov=42):
    # img = cv2.imread(img_path)
    img = cv2.imdecode(np.fromfile(str(img_path)), cv2.IMREAD_UNCHANGED)
    # 扩展图像，保证内容不超出可视范围
    img = cv2.copyMakeBorder(img, 200, 200, 200, 200, cv2.BORDER_CONSTANT, 0)
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

    z = np.sqrt(w ** 2 + h ** 2) / 2 / np.tan(rad(fov / 2))
    # 齐次变换矩阵
    rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(rad(anglex)), -np.sin(rad(anglex)), 0],
                   [0, -np.sin(rad(anglex)), np.cos(rad(anglex)), 0, ],
                   [0, 0, 0, 1]], np.float32)

    ry = np.array([[np.cos(rad(angley)), 0, np.sin(rad(angley)), 0],
                   [0, 1, 0, 0],
                   [-np.sin(rad(angley)), 0, np.cos(rad(angley)), 0, ],
                   [0, 0, 0, 1]], np.float32)

    rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)), 0, 0],
                   [-np.sin(rad(anglez)), np.cos(rad(anglez)), 0, 0],
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


if __name__ == '__main__':
    img_dir = Path(r"E:\qiantupsd\viewable_area\mcbv1")
    json_dir = Path(r"E:\qiantupsd\viewable_area\mcbv1")

    out_img_dir = Path(r"E:\qiantupsd\viewable_area\png_transform")
    out_img_dir.mkdir(exist_ok=True, parents=True)
    out_json_dir = Path(r"E:\qiantupsd\viewable_area\png_transform")
    out_json_dir.mkdir(exist_ok=True, parents=True)

    for img_path in tqdm(img_dir.glob("*.[jp][pn]g")):
        json_path = json_dir.joinpath(img_path.stem + '.json')
        json_file = open(str(json_path), 'r', encoding='UTF-8')
        objs = json.load(json_file)["shapes"]
        vertexes_list = []
        for obj in objs:
            vertexes = np.array(obj["points"], dtype=np.float32).reshape(-1, 2)
            # vertexes += 200
            vertexes_list.append(vertexes)

        # vertexes_dict = {"ret": vertexes_list}
        # out_img_path = out_img_dir.joinpath(img_path.stem + '_ori' + '.jpg')
        # shutil.copy(img_path, out_img_path)
        # out_json_path = out_json_dir.joinpath(img_path.stem + '_ori' + '.json')
        #
        # json_content = json.dumps(vertexes_list, cls=MyEncoder, ensure_ascii=False)  # 将字典转化为字符串，带缩进
        # with open(str(out_json_path), 'w', encoding='UTF-8') as fp:
        #     fp.write(json_content)

        for idx in range(10):
            trans_img, warpR = random_perspect(img_path)
            trans_vertexes_list = []
            # tmp_img = trans_img.copy()
            for vertexes in vertexes_list:
                trans_vertexes = coordinateTransform(vertexes, warpR)
                trans_vertexes_list.append(trans_vertexes)
            # for x_trans, y_trans in trans_vertexes:
            #         cv2.circle(tmp_img, (int(x_trans), int(y_trans)), 5, (0, 255, 0), 5)
            #
            # cv2.imshow("result_after", tmp_img)
            # cv2.waitKey(0)

            trans_vertexes_dict = {"ret": trans_vertexes_list}
            out_img_path = out_img_dir.joinpath(img_path.stem + '_trans_' + str(idx) + '.jpg')
            cv2.imwrite(str(out_img_path), trans_img)

            out_json_path = out_json_dir.joinpath(img_path.stem + '_trans_' + str(idx) + '.json')

            json_content = json.dumps(trans_vertexes_dict, cls=MyEncoder, ensure_ascii=False)  # 将字典转化为字符串，带缩进
            with open(str(out_json_path), 'w', encoding='UTF-8') as fp:
                fp.write(json_content)
