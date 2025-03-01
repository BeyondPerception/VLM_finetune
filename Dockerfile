FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel

RUN pip install flash-attn --no-build-isolation
RUN pip install transformers==4.46.3 accelerate==1.0.1
RUN pip install decord ffmpeg-python imageio opencv-python

ENTRYPOINT "python tune.py"