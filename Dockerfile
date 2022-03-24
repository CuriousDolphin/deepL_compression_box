FROM nvidia/cuda:11.4.2-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

# Install linux packages
#RUN apt update && apt install -y zip htop -y software-properties-common  python3 python3-pip
RUN apt-get -y update \
    && apt-get install -y software-properties-common \
    && apt-get -y update \
    && add-apt-repository universe
RUN apt-get -y install python3  python3-pip zip htop libgl1-mesa-glx


# Install python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
#RUN pip uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof
RUN pip install --no-cache -r requirements.txt 
#RUN pip install --no-cache -U torch torchvision numpy
# RUN pip install --no-cache torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# Create working directory
RUN mkdir -p /app
WORKDIR /app

# Copy contents
COPY . /app

# Set environment variables
ENV HOME=/app
ENTRYPOINT [ "python3" ]