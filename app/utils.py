import os

import matplotlib.pyplot as plt
plt.show()
import numpy as np
import torch
from torch import nn
from torchvision import models
import time
import warnings
warnings.filterwarnings("ignore")
import copy
from tqdm import tqdm

tqdm.monitor_interval = 0  # 禁用监控刷新机制

#展示数据的函数
def im_convert(tensor):
    image = tensor.to('cpu').clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose((1, 2, 0))
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = image.clip(0, 1)
    return image

#迁移学习的函数，不更新参数
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for params in model.parameters():
            params.requires_grad = False #不计算反向传播梯度,不更新参数

#本地修改路径导入模型
def load_model(use_pretrained):
    # 检查官网源，use_pretrained=True自动下载，use_pretrained=False使用本地源
    model_ft = models.resnet152(pretrained=use_pretrained)
    model_cache_path = "./original_model/resnet152-394f9c45.pth"
    if use_pretrained:
        print("Loading Online models")
        model_ft = models.resnet152(pretrained=use_pretrained)
    else:
        print("Loading Local models")
        if not os.path.exists(model_cache_path):
            raise FileNotFoundError(f"预训练模型未找到于 {model_cache_path}")
        # 加载本地权重
        state_dict = torch.load(model_cache_path, map_location='cpu')
        model_ft.load_state_dict(state_dict)
    return model_ft

#设置输出层的函数
def initialize_model(model_name, num_classes, feature_extract, use_pretrained):
    model_ft = load_model(use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc=nn.Linear(num_ftrs, num_classes) # 类别数自己根据自己任务来
    input_size=64 # 输入大小根据自己配置来
    return model_ft, input_size

#训练模型函数
#模型、数据集、损失函数、优化器、迭代次数、保存文件
def train_model(model, scheduler, device, dataloaders, criterion, optimizer, num_epochs=50, filename='best.pt'):
    since = time.time() # 算时间
    best_acc = 0 # 记录最好的那一次
    model.to(device) # 模型也得放到CPU或者GPU
    # 训练过程中打印一堆损失和指标
    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    # 学习率
    LRs = [optimizer.param_groups[0]['lr']]
    # 最好的那次模型，后续会变的，先初始化
    best_model_wts = copy.deepcopy(model.state_dict())
    #epoch循环
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)
        #训练与验证
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            #把数据遍历并显示进度
            # 使用 tqdm 包装数据加载器并设置描述
            pbar = tqdm(dataloaders[phase], desc=f"{phase} Progress", unit="batch", leave=True)
            # 遍历 tqdm 包装后的 dataloader，这样进度条才会显示更新
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                # 动态更新 tqdm 的描述信息
                #pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{(torch.sum(preds == labels.data) / inputs.size(0)):.4f}"})
                current_loss = running_loss / ((pbar.n + 1) * inputs.size(0))
                current_acc = running_corrects.double() / ((pbar.n + 1) * inputs.size(0))
                # 动态更新 tqdm 的描述信息，显示当前 loss 和 accuracy
                pbar.set_postfix({'loss': f"{current_loss:.4f}", 'acc': f"{current_acc:.4f}"})
            pbar.close()
            epoch_loss = running_loss / len(dataloaders[phase].dataset) # 计算平均值
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            time_elapsed = time.time() - since # 计算一共epoch花费多长时间
            print('Time elapsed {:.0f}m{:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            #得到训练最优的一次模型
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                #以字典形式保存
                state = {
                    'state_dict': model.state_dict(),
                    'best_acc': epoch_acc,
                    'optimizer': optimizer.state_dict()
                }
                torch.save(state, filename)
            if phase == 'valid':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)
        print('Optimization learning rate: {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print(' ')
        scheduler.step()
    time_elapsed = time.time() - since #一次训练的时间
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 训练完之后用最好的一次作为最终模型结果
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs