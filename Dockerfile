# 使用 PyTorch 官方映像作為基礎映像
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
# 設置工作目錄
WORKDIR /app
# 複製當前目錄內容到容器內
COPY . .
# 安裝 Flask
RUN pip install flask torch
# 暴露應用的埠
EXPOSE 3000
# 啟動 Flask 應用
CMD ["python", "app.py"]


