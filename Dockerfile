FROM ubuntu:20.04 as builder

FROM python:3.10.11

# Create a user to run the application
RUN apt-get update && \
      apt-get -y install sudo

# Install ffmpeg
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN useradd -ms /bin/bash docker && echo "docker:docker" | chpasswd && adduser docker sudo

# Allow sudo without password (add docker user to sudoers)
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Switch to the docker user
USER docker

# Become root
USER root

# Set the working directory for the container
WORKDIR /home/docker
COPY . .

# Install PIL and tk dependencies
RUN sudo apt-get install apt-utils
RUN sudo apt-get update && apt-get install -y \
    python3-pil \
    python3-tk \
    python3-pil.imagetk

# Install the dependencies
RUN pip install --upgrade pip; \
    pip3 install -r requirements.txt

RUN python3 install.py
    
# Create /.local directory
RUN mkdir /.local; mkdir /.local/share; chmod 777 /.local; chmod 777 /.local/share

# Set the display port to avoid crash and to be able to use the container with a remote host
ENV DISPLAY=:0

# Run the application
CMD ["python3", "main.py"]