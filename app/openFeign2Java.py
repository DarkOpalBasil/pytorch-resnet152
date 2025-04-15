import io
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify, send_file

import utils

# 初始化 Flask 应用
app = Flask(__name__)

# 1. 数据预处理
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 2. 读取类别标签
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# 3. 加载模型
model_name = 'resnet152'  # 使用 ResNet152 与训练模型保持一致
feature_extract = True
train_on_gpu = torch.cuda.is_available()

device = torch.device("cuda:0" if train_on_gpu else "cpu")
if train_on_gpu:
    print('Running on GPU')
else:
    print('Running on CPU')

# 初始化模型，使用 utils 文件的 initialize_model
model_ft, input_size = utils.initialize_model(model_name, 6, feature_extract, use_pretrained=True)
model_ft = model_ft.to(device)

# 加载权重
checkpoint = torch.load('best.pt', map_location=device)
model_ft.load_state_dict(checkpoint['state_dict'])
model_ft.eval()

# 4. 图像预测函数
def predict_image(image_bytes):
    """对单张图片进行预测"""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = data_transforms(image).unsqueeze(0).to(device)  # 加入 batch 维度

        # 预测图像
        with torch.no_grad():
            output = model_ft(image)
            _, preds_tensor = torch.max(output, 1)

        # 转换为 numpy 数据
        preds = np.squeeze(preds_tensor.cpu().numpy()) if train_on_gpu else np.squeeze(preds_tensor.numpy())
        class_id = preds + 1  # +1 因为是从 0 开始

        # 返回预测结果
        class_name = cat_to_name[str(class_id)]
        return class_id, class_name

    except Exception as e:
        return str(e), None

# 5. 生成预测结果可视化
def visualize_prediction(image_bytes, class_name):
    """返回图片和预测名称的可视化图像"""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image)
    ax.axis('off')
    ax.set_title(f"Prediction: {class_name}", color="green")
    # 展示图片
    # plt.show()

    # 保存图像
    output_path = "static/prediction_result.jpg"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path

# 6. Flask 路由：接收图片并进行预测
@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # 读取图片数据
    image_bytes = file.read()

    # 进行预测
    class_id, class_name = predict_image(image_bytes)
    if class_name is None:
        return jsonify({'error': 'Error processing image'}), 500

    # 生成可视化图片
    result_image_path = '/' + visualize_prediction(image_bytes, class_name)

    # 返回 JSON 结果
    result = {
        'class_id': int(class_id),
        'class_name': class_name,
        'result_image_url': result_image_path
    }
    return jsonify(result)

# 7. 静态图片访问接口
@app.route('/static/<path:filename>')
def serve_static(filename):
    """访问静态图片"""
    return send_file(f'project/static/{filename}')

# 8. 启动 Flask 服务
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)