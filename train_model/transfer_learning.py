import os

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

#读取JSON数据，标签对应文件夹
with open('../app/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

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

#再次训练，解除迁移学习限制
for param in model_ft.parameters():
    param.requires_grad = True

#设置输出层
#设置那些层进行训练以及分类数
model_ft, input_size = utils.initialize_model(model_name, 6, feature_extract, use_pretrained=True)
#GPU还是CPU计算
model_ft = model_ft.to(device)

#再训练，学习率调小
optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-3) # Adam优化器，学习率
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=5,gamma=0.1) # 学习率调度器
criterion = nn.CrossEntropyLoss() # 损失函数

#加载之前训练好的权重参数
filename = 'best.pt' # 模型名称
checkpoint = torch.load(filename) #把最好的一次模型加载进来
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])

#再次训练
#此处再次手动进行下一个训练过程时，注意这里的名称可以会因为引用而变化
new_filename = 'transfer_learning_best.pt'
print('\ntransfer_learning: ')
model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = utils.train_model(model_ft, scheduler, device, dataloaders, criterion, optimizer_ft, 50, new_filename)
##最后保存模型为transfer_learning_best.pt