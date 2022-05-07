from importlib.resources import path
from pyexpat import model
import re
from PIL import Image
from matplotlib import image
from model.residual_attention_network import ResidualAttentionModel_92_32input_update as ResidualAttentionModel
from model.residual_attention_network import ResidualModel_92_32input_update as ResNet
from torchvision import transforms
import random
import os
import torch
from flask import render_template, Blueprint, request
import json
import pandas as pd
import numpy as np
import joblib
import time
import gc
import warnings
import base64
import uuid

warnings.filterwarnings("ignore")

app_query = Blueprint('app_query', __name__, template_folder='templates')


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


seed_everything(42)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device('cpu')
model1 = ResidualAttentionModel()  # 初始化模型
model1.to(device)
model2 = ResNet()  # 初始化模型
model2.to(device)


def save_base64_img(img_info):
    ext = img_info.groupdict().get('ext')  # 后缀
    data = img_info.groupdict().get('data')  # 图片data

    image = base64.urlsafe_b64decode(data)
    filename = "{}.{}".format(uuid.uuid4(), ext)  # 文件名
    path = "{}{}".format('./temp_data/', filename)  # 服务器图片存路径
    with open(path, mode='wb') as f:
        f.write(image)
    return path


def test_one_img(model, img_path,  model_file='./best_model.pth'):
    # Test
    transform_valid = transforms.Compose([transforms.Resize((32, 32)),
                                          transforms.ToTensor()])
    images = Image.open(img_path)
    images = transform_valid(images).unsqueeze(0)  # 拓展维度

    model.load_state_dict(torch.load(
        model_file, map_location=torch.device('cpu')))
    model.eval()

    images = images.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    proba = torch.max(torch.softmax(outputs.data, 1), 1)[0]

    return classes[predicted[0]], round(float(proba), 3)


with open('./vis_data.json', encoding='utf-8') as f:
    vis_file = json.load(f)


@app_query.route('/query_page')
def query_page():
    return render_template('query_page.html')


@app_query.route('/get_loss', methods=['POST'])
def get_loss():
    train_loss = vis_file['train_loss']
    eval_loss = vis_file['eval_loss']
    return json.dumps({'train_loss': train_loss, 'eval_loss': eval_loss})


@app_query.route('/predict', methods=['POST'])
def predict():
    image64 = request.form['image']
    img_info = re.search(
        'data:image/(?P<ext>.*?);base64,(?P<data>.*)', image64, re.DOTALL)

    if img_info:
        path = save_base64_img(img_info)
        class_1, proba_1 = test_one_img(model1, path, './best_model.pth')
        class_2, proba_2 = test_one_img(
            model2, path, './best_model_resnet.pth')

        return json.dumps({
            'model1': f'模型1(注意力): 类别 {class_1}  概率 {proba_1}',
            'model2': f'模型2(无注意力): 类别 {class_2}  概率 {proba_2}',
        })
    else:
        raise Exception('Cannot parse!')
