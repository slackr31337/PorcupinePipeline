# Dockerfile
FROM debian:bookworm-slim as builder

ENV DEBIAN_FRONTEND noninteractive
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

ENV VIRTUAL_ENV=/opt/venv

RUN apt-get update -y && apt-get upgrade -y &&\
    apt-get install -y --no-install-recommends \
        gcc python3 python3-dev python3-pip python3-venv \
        python3-gi python3-gi-cairo \
        pkg-config libcairo2 libcairo2-dev \
        gobject-introspection libgirepository1.0-dev && \
    apt-get clean

RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY ./requirements.txt requirements.txt

RUN . $VIRTUAL_ENV/bin/activate &&\
 $VIRTUAL_ENV/bin/pip install --no-cache-dir --upgrade pip setuptools wheel &&\
 $VIRTUAL_ENV/bin/pip install --no-cache-dir -r requirements.txt

FROM debian:bookworm-slim

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3 alsa-utils libsndfile1-dev && \
    apt-get clean

COPY --from=builder $VIRTUAL_ENV $VIRTUAL_ENV

RUN mkdir /app
WORKDIR /app

COPY . .
RUN ls -l && mv docker/entrypoint.sh /entrypoint.sh && chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
