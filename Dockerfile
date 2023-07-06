FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

ARG MODEL_URL='https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/v2-1_768-ema-pruned.ckpt'

ARG HF_TOKEN=''

RUN apt update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install git wget \
    python3.10 python3-pip \
    build-essential libgl-dev libglib2.0-0 vim
RUN ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app


RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git webui

WORKDIR /app/webui

ENV MODEL_URL=${MODEL_URL}
ENV HF_TOKEN=${HF_TOKEN}

RUN pip install tqdm requests

ADD download_checkpoint.py .

RUN python download_checkpoint.py

ADD prepare.py .

RUN python prepare.py --skip-torch-cuda-test --xformers --reinstall-torch --reinstall-xformers

ADD download.py download.py
RUN python download.py --use-cpu=all

ADD app.py app.py

CMD python app.py