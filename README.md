# PyTorch Training Of ResNet152 Project
## python训练残差神经网络ResNet152
> 这是一个基于PyTorch框架，通过对官方ResNet152模型进行参数微调，来实现自己的分类任务。  

该项目实现了玉米叶部病变图像识别的分类任务，训练数据来自多处网络开源数据，此处已经整理成压缩包提供下载。  

此项目的模型与算法可用于任何数据的识别分类，模型也可替换为ResNet18、ResNet50、ResNet101、ResNet152的系列模型。  

还使用了Flask Web框架搭建了一个小型的Web应用，运行在5000端口号上，可以在本地通过 http://localhost:5000/predict 的API调用模型，参数为 ‘POST请求 + form-data’;

Web服务预测结果示意图：  
<img width="300"/>https://github.com/DarkOpalBasil/pytorch-resnet152/data/corn_leaf/test/prediction_result.jpg

### 项目架构:   
```
pythonTroch/                        # 项目根目录
│
├── acc_model/                      # 模型保存目录
│   ├── 80%/                        # 模型准确率
│   │   └── best.pt                 # 模型名称
│   └── original_model/             # 原始模型
│       └── resnet152.pth           # ResNet152官网模型
│
├── app/                            # Web 应用目录(此目录部署Docker)
│   ├── static/                     # Web 静态资源
│   │   └── prediction_result.jpg   # 识别结果图像
│   ├── best.pt                     # 应用模型
│   ├── cat_to_name.json            # 对应疾病名称
│   ├── openFeign2Java.py           # Flask Web 应用
│   └── utils.py                    # 工具类(多个程序使用)                      
│
├── data/                           # 数据目录
│   └── corn_leaf/                  # 玉米叶部图像
│       ├── test/                   # 测试集
│       ├── train/                  # 训练集
│       └── valid/                  # 验证集
│
├── train_model/                    # 训练模型目录
│   ├── best.pt                     # 训练保留最好的模型
│   ├── JSON2Disease.txt            # 对应中文疾病名称
│   ├── test.py                     # 测试程序
│   ├── train.py                    # 训练程序
│   ├── transfer_learning.py        # 迁移学习程序
│   ├── transfer_learning_best.pt   # 迁移学习保留最好的模型
│   └── valid.py                    # 验证程序(有测试程序其实不需要此程序) 
│
├── .git/                           # Git仓库
│
├── .dockerignore                   # Docker忽略规则
├── .gitignore                      # Git忽略规则
├── docker-compose.yml              # Docker容器编排
├── Dockerfile                      # Docker配置文件
├── README.md                       # 项目说明
├── requirements_cpu.txt            # 依赖清单(cpu版本)
└── requirements_gpu.txt            # 依赖清单(gpu版本)
```
* 注意这个Web程序没有处理高并发的功能，多个访问（大概15个以上）Web服务会瘫痪，可以使用消息队列来改进这个缺陷，实现流量削峰。

* 该项目可部署到Docker中，但取决于目标主机的性能，应当手动选择是使用GPU运行的依赖还是CPU运行的依赖。

---
English Version：  
> This is a project based on the PyTorch framework, which fine-tunes the parameters of the official ResNet152 model to achieve a custom classification task.  

The project implements a classification task for corn leaf disease image recognition. The training data comes from multiple open-source datasets on the internet, which have been compiled into a downloadable compressed package.  

The model and algorithm of this project can be used for the recognition and classification of any data, and the model can also be replaced with the ResNet series models such as ResNet18, ResNet50, ResNet101, and ResNet152.  

Additionally, a small web application was built using the Flask web framework, running on port 5000. The model can be invoked locally via the API at http://localhost:5000/predict, with parameters passed as 'POST request + form-data'.  

### Project Structure:   
```
pythonTroch/                        # Project root directory
│
├── acc_model/                      # Model saving directory
│   ├── 80%/                        # Model accuracy
│   │   └── best.pt                 # Model name
│   └── original_model/             # Original model
│       └── resnet152.pth           # ResNet152 official model
│
├── app/                            # Web application directory (this directory is deployed with Docker)
│   ├── static/                     # Web static resources
│   │   └── prediction_result.jpg   # Recognition result image
│   ├── best.pt                     # Application model
│   ├── cat_to_name.json            # Corresponding disease names
│   ├── openFeign2Java.py           # Flask web application
│   └── utils.py                    # Utility class (used by multiple programs)                      
│
├── data/                           # Data directory
│   └── corn_leaf/                  # Corn leaf images
│       ├── test/                   # Test set
│       ├── train/                  # Training set
│       └── valid/                  # Validation set
│
├── train_model/                    # Training model directory
│   ├── best.pt                     # Best model saved during training
│   ├── JSON2Disease.txt            # Corresponding Chinese disease names
│   ├── test.py                     # Testing program
│   ├── train.py                    # Training program
│   ├── transfer_learning.py        # Transfer learning program
│   ├── transfer_learning_best.pt   # Best model saved during transfer learning
│   └── valid.py                    # Validation program (not necessary if there is a testing program) 
│
├── .git/                           # Git repository
│
├── .dockerignore                   # Docker ignore rules
├── .gitignore                      # Git ignore rules
├── docker-compose.yml              # Docker container orchestration
├── Dockerfile                      # Docker configuration file
├── README.md                       # Project description
├── requirements_cpu.txt            # Dependency list (CPU version)
└── requirements_gpu.txt            # Dependency list (GPU version)
```

* Note that this web application does not handle high concurrency. Multiple accesses (approximately 15 or more) will cause the web service to crash. A message queue can be used to improve this flaw and achieve traffic peak shaving.  

* The project can be deployed to Docker, but depending on the performance of the target host, you should manually choose whether to use GPU or CPU dependencies.