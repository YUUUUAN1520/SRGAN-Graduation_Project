#!/usr/bin/env python
# coding: utf-8


from PIL import Image
import numpy as np
from Utils import normalize, denormalize
import tensorflow as tf
import argparse

sub_size = 96
parser = argparse.ArgumentParser()
parser.add_argument('img_path')
parser.add_argument('img_name')
args = parser.parse_args()

###-----------在此更改所需的图片路径-----------###
file_path = args.img_path
image_name = args.img_name
imgs_path = file_path + "\\" +image_name
###------------------------------------------###


model_path = '.\\Trained_model\\gen_model3000.h5'
output_path = '.\\application\\'




#分割图片
downscale_factor = 4
save_percent = 0.6
cut_pixels = int( ( 1.0 - save_percent ) * sub_size / 2 )
cut_trained_pixels = cut_pixels * downscale_factor

def cut_image(img_path):
    image = Image.open(img_path)
    width, height = image.size
    point = [0,0]
    Coordinate_points = []
    count_x = 0
    count_y = 0
    count_flag = False
    
    while(1):
        if(point[0] + sub_size <= width) and (point[1] + sub_size <= height):
            Coordinate_points.append(point.copy())
            point[0] += sub_size - 2 * cut_pixels
        elif(point[0] + sub_size > width) and (point[1] + sub_size <= height):
            if not count_flag:
                count_flag = True
                count_x = len(Coordinate_points)
            point[0] = 0
            point[1] += sub_size - 2* cut_pixels
        else:
            break
    count_y = len(Coordinate_points) // count_x
    box_list = []
    for point in Coordinate_points:
        box = (point[0], point[1], point[0]+sub_size, point[1]+sub_size)
        box_list.append(box)
        
    image_list = [image.crop(box) for box in box_list]
    image_list_array = []
    for img in image_list:
        image = normalize(np.array(img).astype(np.float64))
        image_list_array.append(image)
    
    return (count_x, count_y, image_list_array)

def cut_gen_image(gen_image_list):
    cut_gen_list = []
    for img in gen_image_list:
        cut_gen_list.append(img[:,
                                cut_trained_pixels:-cut_trained_pixels,
                                cut_trained_pixels:-cut_trained_pixels,
                                :])
    return cut_gen_list





def load_generator(path):
    Gen = tf.keras.models.load_model(path, compile=False)
    return Gen

def gen_new_one(model_path, imgs_path, output_path):
    gen = load_generator(model_path)
    x, y, image_list = cut_image(imgs_path)

    gen = load_generator(model_path)
    image_gen_list = []
    for i in range(len(image_list)):
        image_gen_list.append(denormalize(gen.predict(np.array([image_list[i]]))))

    cut_gen_list = cut_gen_image(image_gen_list)

    rows = []
    for j in range(y):
        row = cut_gen_list[j*x]
        for i in range(1,x):
            row = np.concatenate((row,cut_gen_list[j*x+i]),axis=2)
        rows.append(row)
    column = rows[0]
    for j in range(1,y):
        column = np.concatenate((column,rows[j]),axis=1)
    
    Image.fromarray(np.uint8(column[0])).save(output_path +'gen_' + image_name)





if __name__ == "__main__":
    gen_new_one(model_path, imgs_path, output_path)
    print('您选择的 {} 已成功生成完成！'.format(image_name))







