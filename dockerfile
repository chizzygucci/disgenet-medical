FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --prefer-binary --default-timeout=1000 --retries=10 --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
COPY . .
CMD ["uvicorn", "inferenceserve:app", "--host", "0.0.0.0", "--port", "8000"]



