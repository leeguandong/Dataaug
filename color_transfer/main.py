import cv2
import numpy as np
from pathlib import Path
from skimage import color
from matplotlib.colors import rgb2hex, hex2color, rgb_to_hsv
import math
import time
from color_transfer import color_transfer
from PIL import Image

if __name__ == '__main__':
    tic = time.time()
    # img_path = Path('未裁剪素材/621_1242_1/allbg.png')
    # img_path = Path('text_input.png') # 6646DD
    img_path = Path('input1_8C5EDF.png')  # input1_8C5EDF
    # img_path = Path('input2_FC6B0f.png')  # input2_FC6B0f
    # img_path = Path('input3_FA6C6F.png')  # input3_FA6C6F

    cv_img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), -1)
    out_img = color_transfer(cv_img, '#00ffff', '#ffd312')  # target_color, base_color
    tar_c_list = ['#0090FF', '#E230F2', '#762BF9', '#ff7ea7', '#FF4E41', '#bea9ff', '#ffd312']
    # tar_c_list = ['#ffd312'] #ffd30c
    # tar_c_list = ['#6646DD']
    # tar_c_list = ['#ffd30c']
    for tar_color in tar_c_list:
        out_img = color_transfer(cv_img, tar_color, '#8C5EDF')  # target_color, base_color  #1FE29D, 6646DD
        cv2.imencode('.png', out_img)[1].tofile(tar_color + '_out1.png')
    print(time.time() - tic, "-----------------------------")

    tic = time.time()
    img_path = Path('input2_FC6B0f.png')  # input2_FC6B0f
    cv_img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), -1)
    tar_c_list = ['#0090FF', '#E230F2', '#762BF9', '#ff7ea7', '#FF4E41', '#bea9ff', '#ffd312']
    for tar_color in tar_c_list:
        out_img = color_transfer(cv_img, tar_color, '#FC6B0f')  # target_color, base_color  #1FE29D, 6646DD
        cv2.imencode('.png', out_img)[1].tofile(tar_color + '_out2.png')
    print(time.time() - tic, "-----------------------------")

    tic = time.time()
    img_path = Path('input3_FA6C6F.png')  # input3_FA6C6F
    cv_img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), -1)
    tar_c_list = ['#0090FF', '#E230F2', '#762BF9', '#ff7ea7', '#FF4E41', '#bea9ff', '#ffd312']
    for tar_color in tar_c_list:
        out_img = color_transfer(cv_img, tar_color, '#FA6C6F')  # target_color, base_color  #1FE29D, 6646DD
        cv2.imencode('.png', out_img)[1].tofile(tar_color + '_out3.png')
    print(time.time() - tic, "-----------------------------")

    tic = time.time()
    img_path = Path('text_input.png')  # input2_FC6B0f
    cv_img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), -1)
    tar_c_list = ['#0090FF', '#E230F2', '#762BF9', '#ff7ea7', '#FF4E41', '#bea9ff', '#ffd312']
    for tar_color in tar_c_list:
        out_img = color_transfer(cv_img, tar_color, '#6646DD')  # target_color, base_color  #1FE29D, 6646DD
        cv2.imencode('.png', out_img)[1].tofile("_" + tar_color + '_out.png')
    print(time.time() - tic, "-----------------------------")

    tic = time.time()
    img_path = Path('input4.png')  # input3_FA6C6F
    cv_img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), -1)
    tar_c_list = ['#0090FF', '#E230F2', '#762BF9', '#ff7ea7', '#FF4E41', '#bea9ff', '#ffd312']
    for tar_color in tar_c_list:
        out_img = color_transfer(cv_img, tar_color)  # target_color, base_color  #1FE29D, 6646DD , FDFEFE
        cv2.imencode('.png', out_img)[1].tofile(tar_color + '_out4.png')
    print(time.time() - tic, "-----------------------------")
