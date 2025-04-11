# 使用官方 Python 基础镜像
FROM python:3.11

# 设置工作目录
WORKDIR /app

# 将 Python 脚本复制到容器中
COPY . /app

# 将项目的依赖文件复制到工作目录
COPY requirements_cpu.txt /app

# 安装依赖
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements_cpu.txt

# 暴露应用的端口
EXPOSE 5000

# 运行 Python 应用
CMD ["python", "openFeign2Java.py"]