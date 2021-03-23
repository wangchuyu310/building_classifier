FROM python:3.6-alpine
EXPOSE 5000
RUN pip install flask
RUN mkdir -p /app/src
WORKDIR /app
ADD . /app
CMD ["python", "/app/src/index.py"]