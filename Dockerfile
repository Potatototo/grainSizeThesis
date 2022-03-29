FROM ubuntu:latest

RUN apt-get update && apt-get install -y sudo git python3-pip python3-dev

RUN pip3 install numpy opencv-python pandas ray ray[tune]
