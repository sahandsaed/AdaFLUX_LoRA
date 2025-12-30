FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y git wget build-essential

# Create workdir
WORKDIR /workspace/AdaFLUX-LoRA

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src ./src
COPY config.yaml .
COPY run.sh .

# TensorBoard port
EXPOSE 6006

CMD ["bash", "run.sh"]
