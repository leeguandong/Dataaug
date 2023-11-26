import os
import pathlib
from PIL import Image

# anno
path_anno = r'E:\qiantupsd\viewable_area\mcbv1'

for anno in os.listdir(path_anno):
    if anno.split('.j')[-1] == "son":
        name = anno.split('.')[0] + '.jpg'
        img = os.path.join(path_anno, name)
        path_txt = pathlib.Path(img)
        # if path_txt.exists():
        #     pass
        try:
            image = Image.open(img)
        except:
            print(anno)
            os.remove(os.path.join(path_anno, name))
