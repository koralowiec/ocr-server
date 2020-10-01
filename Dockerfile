FROM ubuntu:focal as prod

LABEL org.opencontainers.image.source https://github.com/koralowiec/ocr-server

WORKDIR /src

# https://serverfault.com/a/1016972
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Warsaw

# https://github.com/conda-forge/pygridgen-feedstock/issues/10#issuecomment-365914605
RUN apt update && \
    apt install python3 python3-pip tesseract-ocr libgl1-mesa-glx -y 

COPY requirements.txt ./

RUN pip3 install -r requirements.txt

COPY . ./

CMD [ "uvicorn", "--app-dir", "src", "main:app", "--port", "5000", "--host", "0.0.0.0" ]

# for local development
FROM prod as dev

CMD [ "uvicorn", "--app-dir", "src", "main:app", "--reload", "--port", "5000", "--host", "0.0.0.0" ]