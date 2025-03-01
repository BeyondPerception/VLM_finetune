FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel

RUN pip install "/volume/flash_attn-2.7.2.post1+cu11torch2.1cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
RUN pip install transformers==4.46.3 accelerate==1.0.1
RUN pip install decord ffmpeg-python imageio opencv-python

CMD "python tune.py"