FROM continuumio/anaconda

RUN conda install flask
RUN conda install -c https://conda.binstar.org/menpo opencv
RUN apt-get update
RUN apt-get install -y \
  libgtk2.0-0
