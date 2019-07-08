# The first instruction is what image we want to base our container on
# We Use an official Python runtime as a parent image
#FROM python:3.-alpine
 
 # The first instruction is what image we want to base our container on
# We Use an official Python runtime as a parent image
FROM python:3.6.6-stretch

# The enviroment variable ensures that the python output is set straight
# to the terminal with out buffering it first
ENV PYTHONUNBUFFERED 1

ADD . /web_service

#RUN apt-get update && apt-get update && apt-get install -y gcc build-essential autoconf cmake libtool git python-dev apt-utils python-pip libpython-dev
# RUN apk add --update alpine-sdk

 RUN python -m pip install --upgrade pip

RUN pip install awscli


#RUN mkdir /mybin && echo '/mybin directory created'
#ENV PATH="/mybin:${PATH}"
#RUN cd /mybin && ln -sf $(which gcc) x86_64-linux-gnu-gcc && x86_64-linux-gnu-gcc -v
RUN pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.13.1-py3-none-any.whl

#RUN python -m nltk.downloader wordnet 
RUN python -m pip install --upgrade tensorflow
RUN pip install -r web_service/requirements.txt
RUN [ "python", "-c", "import nltk; nltk.download('wordnet')" ]

#RUN python -m spacy download en_core_web_md
#RUN python -m nltk.downloader stopwords

EXPOSE 80

CMD ["python", "web_service/manage.py", "runserver", "0.0.0.0:80"]


