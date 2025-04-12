import os

import matplotlib.pyplot as plt

plt.show()
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms, models, datasets
import warnings
warnings.filterwarnings("ignore")
import json
from app import utils

#数据导入
data_dir = '../data/corn_leaf/'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

#数据预处理
data_transforms = {
    'train': transforms.Compose([transforms.RandomRotation(45),
                                 transforms.CenterCrop(224),
                                 transforms.RandomHorizontalFlip(p=0.5),
                                 transforms.RandomVerticalFlip(p=0.5),
                                 transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),#均差标准差使用官方标准
                                 transforms.RandomGrayscale(p=0.025),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                 ]),
    'valid': transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                 ]),
}

#构建训练数据集
batch_size = 8 #输入大小，占用显存很厉害

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform=data_transforms[x]) for x in ['train', 'valid']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes

#可打印查看
# print('原始数据：')
# print(image_datasets)
# print(image_datasets['valid'].transform)  # 确保 valid 集的 transform 不是 None
# print(dataloaders)
# print(dataset_sizes)


#读取JSON数据，标签对应文件夹
with open('../app/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
#可打印查看
print('json标签：')
print(cat_to_name)

#展示部分数据
fig = plt.figure(figsize=(20, 12))
columns = 4
rows = 2
dataiter = iter(dataloaders['valid'])
inputs, classes = next(dataiter)
for i in range(columns*rows):
    ax = fig.add_subplot(rows, columns, i + 1, xticks=[], yticks=[])
    ax.set_title(cat_to_name[str(int(class_names[classes[i]]))])
    plt.imshow(utils.im_convert(inputs[i]))
plt.show()

#模型选择18、50、101、152
model_name = 'resnet152'
feature_extract = True #使用预训练模型，迁移学习，用默认特征
#是否使用GPU训练
train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
    print('Training on GPU')
else:
    print('CUDA is not available, please training on CPU')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#设置模型，152层resnet
model_ft = models.resnet152()
#查看模型参数
print('模型参数：')
print(model_ft)
print('-' * 10)

#可打印查看每层参数设置
print('每层参数：')
for name, param in model_ft.named_parameters():
    print(name)
    print(param)
print('-' * 10)

#！！！调试模型！！！
#设置输出层
#设置那些层进行训练和分类数
model_ft, input_size = utils.initialize_model(model_name, 6, feature_extract, use_pretrained=True)
#GPU还是CPU计算
model_ft = model_ft.to(device)
#是否训练所有层
params_to_update = model_ft.parameters()
print("Params to learn: "+ str(params_to_update)) # 显示默认参数
print('-' * 10)

print('feature_extract: ')
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad:
            params_to_update. append (param)
            print("\t",name)
else:
    for name, param in model_ft.named_parameters():
            if param.requires_grad:
                print("\t",name)
print('-' * 10)
print('\n')

#优化器设置
optimizer_ft=optim.Adam(params_to_update,lr=1e-2) # Adam优化器，学习率
scheduler=optim.lr_scheduler.StepLR(optimizer_ft, step_size=10,gamma=0.1) # 学习率调度器，学习率每10个epoch衰减成原来的1/10
criterion = nn.CrossEntropyLoss() # 损失函数

#！！！定义训练模型的函数！！！
#模型、数据集、损失函数、优化器、迭代次数、保存文件
####！！！开始训练！！！####
model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = utils.train_model(model_ft, scheduler, device, dataloaders, criterion, optimizer_ft, 30, 'best.pt')
#最后保存模型为best.pt