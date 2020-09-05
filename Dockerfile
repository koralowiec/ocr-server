FROM ubuntu:focal as prod

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

ENV FLASK_APP="src/main.py"

CMD [ "flask", "run", "--host=0.0.0.0" ]

# for local development
FROM prod as dev

RUN apt install -y curl \
    && curl -sL https://deb.nodesource.com/setup_12.x | bash \
    && apt install -y nodejs

RUN npm i nodemon -g

CMD [ "nodemon" ]