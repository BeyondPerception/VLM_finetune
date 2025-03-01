FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel

RUN apt update && apt install -y git

RUN git clone https://github.com/DAMO-NLP-SG/VideoLLaMA2 && \
    cd VideoLLaMA2 && \
    pip install --upgrade pip && \
    pip install -e . && \
    pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu118torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"


CMD "/bin/bash 'sleep 2d'"