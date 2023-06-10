# Dockerfile
FROM python:3.10.12-slim-buster as builder

ENV DEBIAN_FRONTEND noninteractive
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

ENV VIRTUAL_ENV=/opt/venv

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        gcc python3-venv && \
    apt-get clean

RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY ./requirements.txt requirements.txt

RUN . $VIRTUAL_ENV/bin/activate &&\
 $VIRTUAL_ENV/bin/pip install --upgrade pip &&\
 $VIRTUAL_ENV/bin/pip install --upgrade wheel &&\
 $VIRTUAL_ENV/bin/pip install -r requirements.txt


FROM python:3.10.12-slim-buster
COPY --from=builder $VIRTUAL_ENV $VIRTUAL_ENV

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    alsa-utils libsndfile1-dev && \
    apt-get clean

RUN echo "deb http://ftp.debian.org/debian sid main" >> /etc/apt/sources.list &&\
    apt-get update && apt-get -y --no-install-recommends -t sid install libc6 libc6-dev libc6-dbg

RUN mkdir /app
WORKDIR /app

COPY . .
RUN ls -l && mv docker/entrypoint.sh /entrypoint.sh && chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
