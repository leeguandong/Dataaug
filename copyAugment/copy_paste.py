import os
import cv2
import random
import background_generator
from PIL import Image
import numpy as np


def create_an_image(bground_path, width, height):
    bground_list = os.listdir(bground_path)
    bground_choice = random.choice(bground_list)
    bground = Image.open(bground_path + bground_choice)
    x, y = random.randint(0, bground.size[0] - width), random.randint(0, bground.size[1] - height)
    bground = bground.crop((x, y, x + width, y + height))
    return bground


# 随机选取 mcb-2 贴合起始的坐标, 根据背景的尺寸和字体的大小选择
def random_x_y(bground_size, font_size):
    width, height = bground_size

    # 为防止文字溢出图片，x，y要预留宽高
    x = random.randint(0, width - font_size * 10)
    y = random.randint(0, int((height - font_size) / 4))

    return x, y


def margins(margin):
    margins = margin.split(",")
    if len(margins) == 1:
        return [margins[0]] * 4
    return [int(m) for m in margins]


def copy_paste(img, background_type, alignment):
    # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
    # img.show()
    width, height = img.size
    # bg_image = create_an_image('./background/', width, height)
    # draw_x, draw_y = random_x_y(bg_image, )

    # margins = margins(5, 5, 5, 5)
    margins = [20, 20, 20, 20]
    margin_top, margin_left, margin_bottom, margin_right = margins
    horizontal_margin = margin_left + margin_right
    vertical_margin = margin_top + margin_bottom
    background_width = width + horizontal_margin
    background_height = height + vertical_margin

    # Generate background image #
    if background_type == 0:
        background_img = background_generator.gaussian_noise(background_height, background_width)
    elif background_type == 1:
        background_img = background_generator.plain_white(background_height, background_width)
    elif background_type == 2:
        background_img = background_generator.quasicrystal(background_height, background_width)
    elif background_type == 3:
        background_img = background_generator.picture(background_height, background_width)
    # background_img.show()

    img = img.convert("RGBA")
    # Place text with alignment #
    if alignment == 0 or width == -1:
        coord = (margin_left, margin_top)
        background_img.paste(img, coord, img)
    elif alignment == 1:
        coord = (int(background_width / 2 - width / 2), margin_top)
        background_img.paste(img, coord, img)
    elif alignment == 2:
        coord = (background_width - width - margin_right, margin_top)
        background_img.paste(img, coord, img)
    # background_img.show()

    # Apply gaussian blur #
    # gaussian_filter = ImageFilter.GaussianBlur(
    #     radius=blur if not random_blur else rnd.randint(0, blur)
    # )
    # final_image = background_img.filter(gaussian_filter)
    # final_mask = background_mask.filter(gaussian_filter)

    img = cv2.cvtColor(np.asarray(background_img), cv2.COLOR_RGB2BGR)
    return img, coord
