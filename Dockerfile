FROM python:3.8
COPY . /MLOps_HW1_new
WORKDIR /MLOps_HW1_new

#ENV PYTHONDONTWRITEBYTECODE 1
#ENV PYTHONUNBUFFERED 1

# RUN pip freeze > requirements.txt
RUN pip install -r requirements.txt
EXPOSE 5000
ENTRYPOINT FLASK_APP=app.py flask run --host=0.0.0.0