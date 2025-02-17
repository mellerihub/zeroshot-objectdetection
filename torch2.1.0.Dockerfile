FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    apt-utils \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa
 
# Install Python 3.10 and pip
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    wget \
    build-essential \
    ca-certificates \
    git \
    gcc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libhdf5-dev \
    && wget https://bootstrap.pypa.io/get-pip.py \
    && python3.10 get-pip.py \
    && rm get-pip.py \
    && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH /usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME /usr/local/cuda-12.1
 
# Specify encoding
ENV LC_ALL=C.UTF-8
 
# Set some environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
 
ENV PATH="/framework:${PATH}"
 
# Set up the program in the image
COPY /.register_source /framework
 
WORKDIR /framework
 
# install requirements
RUN pip3 install --no-cache-dir -r alolib/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
site_packages_location
 
CMD ["python3.10", "main.py"]
