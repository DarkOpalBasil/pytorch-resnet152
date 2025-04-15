import os

import matplotlib.pyplot as plt

plt.show()
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import warnings
warnings.filterwarnings("ignore")
import json
from app import utils

#数据导入
data_dir = '../data/corn_leaf/'
valid_dir = data_dir + '/valid'

#数据预处理
data_transforms = {
    'valid': transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                 ]),
}

#构建训练数据集
batch_size = 8 #输入图片多少，多了占用显存很厉害

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform=data_transforms[x]) for x in ['valid']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['valid']}

#读取JSON数据，标签对应文件夹
with open('../app/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# 模型选择与训练一样
model_name = 'resnet152'
feature_extract = True #使用预训练模型，迁移学习，用默认特征

# 是否使用GPU训练
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print('Training on GPU')
else:
    print('CUDA is not available, please training on CPU')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载训练好的模型
model_ft, input_size = utils.initialize_model(model_name, 6, feature_extract, use_pretrained=True)
# GPU模式
model_ft = model_ft.to(device)
# 加载模型
checkpoint = torch.load('best.pt')
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])

#验证集数据处理
dataiter = iter(dataloaders['valid'])
images, labels = next(dataiter)
model_ft.eval()

if train_on_gpu:
    output = model_ft(images.cuda())
else:
    output = model_ft(images)
#查看验证集大小
print(output.shape)

_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())
reals = labels.numpy()
print('预测值：')
print(preds + 1)
print('真实值：')
print(reals + 1)

fig = plt.figure(figsize=(20, 20))
columns =4
rows = 2
#绘图+名称显示
for idx in range (columns*rows):
    ax = fig.add_subplot (rows, columns, idx+1, xticks=[], yticks=[])
    plt.imshow(utils.im_convert(images[idx]))
    ax.set_title("{} ({})".format(cat_to_name[str(preds[idx].item()+1)], cat_to_name[str(labels[idx].item()+1)]),
                 color=("green" if cat_to_name[str(preds[idx].item()+1)]==cat_to_name[str(labels[idx].item()+1)] else "red"))
plt.show()