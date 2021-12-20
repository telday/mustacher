FROM python:3.10

COPY . /mustache
EXPOSE 5000

RUN apt update
RUN apt install ffmpeg libsm6 libxext6 -y

WORKDIR /mustache
RUN pip install opencv-python flask data_url gunicorn
RUN pip install -e ./python_module/

#ENV FLASK_APP api/mustache_api.py
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

CMD cd api && gunicorn -w 1 --bind 0.0.0.0:5000 wsgi:app
