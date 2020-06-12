# for training on a gpu
# FROM nvidia/cuda:latest  
FROM ubuntu:18.04


LABEL maintainer="Adam Levin <awl92@cornell.edu>"

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN apt-get update
RUN apt-get install -y libatlas-base-dev gfortran nginx supervisor
RUN apt update
RUN apt-get install -y python3-pip libsm6 libxext6 libxrender-dev

RUN mkdir project
COPY requirements.txt /project/requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install -r /project/requirements.txt

RUN useradd nginx

RUN rm /etc/nginx/sites-enabled/default
RUN rm /etc/nginx/sites-available/default
RUN rm /usr/share/nginx/html/index.html

COPY nginx.conf /etc/nginx/
COPY site-nginx.conf /etc/nginx/conf.d/
COPY uwsgi.ini /etc/uwsgi/
COPY supervisord.conf /etc/

COPY app project/app

RUN chown -R nginx:nginx /project/app

ENV TORCH_HOME /project/app

WORKDIR /project

CMD ["/usr/bin/supervisord","-c","/etc/supervisord.conf"]
