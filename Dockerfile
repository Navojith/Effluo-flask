FROM python:latest

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

RUN pip install flask celery redis flask-sqlalchemy flask-wtf

WORKDIR /appFROM python:latest

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED 1

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE 1

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . /app/
