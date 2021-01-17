FROM ubuntu:latest
RUN apt-get update && apt-get -y update
RUN apt-get install -y build-essential python3.6 python3-pip python3-dev
RUN pip3 -q install pip --upgrade

RUN mkdir /usr/src/skin-lesion-model

WORKDIR /usr/src/skin-lesion-model
ADD ./rawdata/ /usr/src/rawdata
ADD ./requirements.txt .
COPY ./Generator/ .

RUN pip3 install -r requirements.txt