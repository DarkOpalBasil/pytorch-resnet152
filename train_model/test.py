import os

import matplotlib.pyplot as plt

plt.show()
import numpy as np
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import warnings
warnings.filterwarnings("ignore")
import json
from app import utils

#数据导入
data_dir = '../data/corn_leaf/'
test_dir = data_dir + '/test'

#数据预处理
data_transforms = {
    'test': transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                 ]),
}

#构建训练数据集
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform=data_transforms[x]) for x in ['test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=len(image_datasets[x]), shuffle=False) for x in ['test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}

#读取JSON数据，标签对应文件夹
with open('../app/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

#模型选择与训练一样
model_name = 'resnet152'
feature_extract = True #使用预训练模型，迁移学习，用默认特征

#是否使用GPU训练
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print('Training on GPU')
else:
    print('CUDA is not available, please training on CPU')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#加载训练好的模型
model_ft, input_size = utils.initialize_model(model_name, 6, feature_extract, use_pretrained=True)
# GPU模式
model_ft = model_ft.to(device)
#加载模型
checkpoint = torch.load('best.pt')
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])

#验证集数据处理
dataiter = iter(dataloaders['test'])
images, labels = next(dataiter)
model_ft.eval()

if train_on_gpu:
    output = model_ft(images.cuda())
else:
    output = model_ft(images)
#查看测试集大小
print(output.shape)

print("每张图像的 Softmax 概率数组：")
for n in range(output.size(0)):
    logits = output[n]  # 取第n个样本的输出
    probs = F.softmax(logits, dim=0)  # 做Softmax归一化
    probs_np = probs.detach().cpu().numpy() if train_on_gpu else probs.detach().numpy()

    # 输出格式化打印
    print(f"样本 {n + 1} 的概率分布: {', '.join([f'{p:.4f}' for p in probs_np])}")
    pred_class = np.argmax(probs_np)
    pred_prob = probs_np[pred_class]
    print(f"预测类别索引: {pred_class}；概率: {pred_prob:.4f}（对应类别: {cat_to_name[str(pred_class + 1)]}）\n")

_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())
# 防止单个数据无法通过索引获取数据
if preds.size == 1:
    preds = np.array([preds])
predsInfo = preds + 1
print('预测值：')
print(predsInfo)

# 动态计算行数
num_images = len(images)  # 获取图像数量
rows = num_images  # 每张图片占一行
# 创建画布
fig = plt.figure(figsize=(5, rows*2))
# 绘图+名称显示
for idx in range(num_images):
    ax = fig.add_subplot(rows, 1, idx + 1, xticks=[], yticks=[])
    plt.imshow(utils.im_convert(images[idx]))
    ax.set_title("Prediction: {}".format(cat_to_name[str(preds[idx].item() + 1)]),
                 color="green")  # 直接显示预测名称
plt.tight_layout()  # 调整布局
plt.show()