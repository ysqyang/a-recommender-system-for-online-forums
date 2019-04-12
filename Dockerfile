FROM centos:7

MAINTAINER Yang Qiu

# base
RUN yum -y update
RUN yum groupinstall -y "Development tools"
RUN yum install -y wget
RUN yum install -y git
RUN yum install -y zsh
RUN yum install -y vim

# Python 3.7
RUN yum install -y gcc openssl-devel bzip2-devel libffi-devel
RUN cd /usr/src
RUN wget https://www.python.org/ftp/python/3.7.3/Python-3.7.3.tgz
RUN tar xzf Python-3.7.3.tgz
RUN Python-3.7.3/configure --enable-optimizations
RUN make altinstall
RUN python3.7 -m ensurepip

# oh my zsh
RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || echo hi

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