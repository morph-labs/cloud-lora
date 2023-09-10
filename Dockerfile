FROM ubuntu:jammy

WORKDIR /root

RUN apt-get update
RUN apt-get install -y git build-essential

RUN git clone https://github.com/ggerganov/llama.cpp
WORKDIR /root/llama.cpp
RUN make

RUN apt-get install -y python3 python3-pip
RUN pip3 install -r requirements.txt
# For the LoRA conversion script 
RUN pip3 install torch --extra-index-url https://download.pytorch.org/whl/cpu

WORKDIR /root

# COPY ggml-model-f16.gguf /root
COPY ggml-model-Q4_0.gguf /root

RUN pip3 install git+https://github.com/abetlen/llama-cpp-python.git@bf08d1b2bb778d5b7

# For the HTTP server
RUN pip3 install fastapi uvicorn python-multipart google-cloud-storage
COPY run.py /root
ENTRYPOINT python3 run.py
