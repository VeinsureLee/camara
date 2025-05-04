# 使用官方 Python 镜像
FROM python:3.10

# 设置容器内的工作目录
WORKDIR /app

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install -r requirements.txt

# 复制项目所有文件
COPY . .

# 运行主程序
CMD ["python", "main.py"]
