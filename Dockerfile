FROM ubuntu:16.04

# base
RUN apt-get update --fix-missing  && apt-get install -y sudo && apt-get install -y apt-utils
RUN apt-get install -y wget && apt-get install -y --no-install-recommends curl
RUN apt-get install -y zsh && apt-get install -y git-core
RUN apt-get install -y vim
RUN apt-get install -y inotify-tools
RUN apt-get update && apt-get install -y locales && rm -rf /var/lib/apt/lists/* && localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8

ADD . /recommender
WORKDIR /code
RUN pip install -r requirements.txt
CMD ["python", "run.py"]