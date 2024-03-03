import numpy as np
from PIL import Image
import imageio

def split_images_from_name(path):
    cubemap = np.array(Image.open(path))
    face_ids = ['negx.png', 'negy.png', 'negz.png', 'posx.png', 'posy.png', 'posz.png']
    cubemap_height, cubemap_width, channels = cubemap.shape
    height = cubemap_height//3
    width = height
    print(cubemap.shape)
    print(height, width)
    vec_img = []
    vec_img.append(cubemap[width:width*2, 0:width])
    vec_img.append(np.rot90(cubemap[0:width, width:width*2], axes=(0,1)))
    vec_img.append(cubemap[width:width*2, width*3:width*4])
    vec_img.append(cubemap[width:width*2, width*2:width*3])
    vec_img.append(np.rot90(cubemap[2*width:width*3, width:width*2], axes=(1,0)))
    vec_img.append(cubemap[width:width*2, width:width*2])
    
    for img, face_id in zip(vec_img, face_ids):
        imageio.imwrite(
            face_id, img)
    
path = 'E:/jorg/phd/visionTransformer/activeLearningLoop/2D_images/P55_0deec78e66a7fce81be8da73219b8a7b.png'
split_images_from_name(path)
