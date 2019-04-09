FROM ubuntu:18.04

MAINTAINER Yang Qiu

# base
RUN apt-get update
RUN apt-get install -y sudo
RUN apt-get install -y apt-utils
RUN apt-get install -y wget
RUN apt-get install -y --no-install-recommends curl
RUN apt-get install -y git
RUN apt-get install -y zsh
RUN apt-get install -y vim
RUN apt-get install -y inotify-tools

# Python 3.7
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y python3.7
RUN apt-get install -y python3-pip

# oh my zsh
RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || echo hi
RUN chsh -s `which zsh` && wget https://raw.githubusercontent.com/ArthurJiang/config/master/.zshrc -O ~/.zshrc
RUN apt-get -y upgrade

# rabbitmq
RUN wget https://packages.erlang-solutions.com/erlang-solutions_1.0_all.deb
RUN dpkg -i erlang-solutions_1.0_all.deb
RUN apt-get install -y erlang erlang-nox
RUN echo 'deb http://www.rabbitmq.com/debian/ testing main' | sudo tee /etc/apt/sources.list.d/rabbitmq.list
RUN wget -O- https://www.rabbitmq.com/rabbitmq-release-signing-key.asc | sudo apt-key add -
RUN apt-get install -y rabbitmq-server
RUN update-rc.d rabbitmq-server defaults

# Chinese support
ENV LANG C.UTF-8

# aliases
RUN echo "alias py='python3.7'" >> ~/.zshrc
RUN echo "alias run='python3.7 /recommender/source/run.py'" >> ~/.zshrc
RUN echo "alias peek='python3.7 /recommender/source/peek.py'" >> ~/.zshrc
RUN echo "alias produce='python3.7 /recommender/source/produce.py'" >> ~/.zshrc
RUN echo "alias test='python3.7 /recommender/source/test.py'" >> ~/.zshrc
RUN echo "alias mq='service rabbitmq-server start'" >> ~/.zshrc

RUN mkdir /recommender
COPY requirements.txt /recommender/
WORKDIR /recommender
RUN python3.7 -m pip install -r requirements.txt
COPY . /recommender
CMD ["zsh"]