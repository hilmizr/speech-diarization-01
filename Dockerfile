FROM python:3.9-slim

WORKDIR /app

RUN mkdir -p /.streamlit && chmod 777 /.streamlit
RUN mkdir -p /tmp && chmod 777 /tmp
RUN mkdir -p /tmp/huggingface && chmod 777 /tmp/huggingface

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    ffmpeg \  
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
COPY src/ ./src/

RUN pip3 install -r requirements.txt

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENV XDG_CACHE_HOME=/tmp
ENV HF_HOME=/tmp/huggingface
RUN mkdir -p /tmp/huggingface && chmod 777 /tmp/huggingface

ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
