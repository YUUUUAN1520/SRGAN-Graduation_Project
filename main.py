import os, json
from flask import Flask, request, send_file
from flask_cors import CORS

from Utils import normalize, plot_generated_images, denormalize
import numpy as np
import imageio
from skimage.transform import resize
import tensorflow as tf

img_shape = (384,384,3)
downscale_factor = 4
h, w = img_shape[0], img_shape[1]
img_res = (h, w)        
low_res = (h // downscale_factor, w // downscale_factor) 

model_path = '.\\Trained_model\\gen_model3000.h5'
output_path = '.\\downloads\\'
processed_img_name = 'gen_img_test.png'

app = Flask(__name__)
# 跨域
CORS(app, supports_credentials=True)


@app.route('/', methods=['POST'])
def index():
    # 定义上传文件保存目录
    upload_path = './uploads'
    file = request.files.getlist('file')[0]
    img_path = os.path.join(upload_path, file.filename)
    file.seek(0)
    # 保存文件
    file.save(img_path)

    # TODO
    # 此处添加调用图片出来方法
    def process_img(img_path):
        img = imageio.imread(img_path).astype(np.float64)
        img_hr = resize(img,img_res)
        img_lr = resize(img,low_res)
        img_hr = normalize(np.array([img_hr]))
        img_lr = normalize(np.array([img_lr]))
        return (img_hr, img_lr)

    def load_generator(path):
        Gen = tf.keras.models.load_model(path, compile=False)
        return Gen

    def test_plot(model_path, img_path, output_path):
        gen = load_generator(model_path)
        (img_hr,img_lr) = process_img(img_path)
        plot_generated_images(output_path, -1, gen, img_hr, img_lr)


    test_plot(model_path, img_path, output_path)

    # 以 json 格式返回结果
    result = {
        'msg': 'ok',
        'code': '200',
        'data': output_path + processed_img_name
    }
    return json.dumps(result)

@app.route('/get_file')
def get_file():
    path = output_path + processed_img_name
    return send_file(path)

if __name__ == '__main__':
    app.run()
