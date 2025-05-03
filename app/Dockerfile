# 使用官方 Python 基础镜像
FROM python:3.11

# 设置工作目录
WORKDIR /app

# 将文件复制到容器中
COPY app/openFeign2Java.py ./
COPY app/best.pt ./
COPY app/cat_to_name.json ./
COPY app/utils.py ./
COPY app/static ./static
COPY app/original_model ./original_model
# 将项目的依赖文件复制到工作目录
COPY ../requirements_cpu.txt .

# 安装依赖
RUN pip install --upgrade pip \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://mirrors.aliyun.com/pypi/simple/ \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    --trusted-host mirrors.aliyun.com \
    --default-timeout=300 \
    -r requirements_cpu.txt

# 暴露应用的端口
EXPOSE 5000

# 运行 Python 应用
CMD ["python", "-u", "openFeign2Java.py"]