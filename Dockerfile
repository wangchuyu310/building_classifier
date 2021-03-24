FROM python:3.6
RUN mkdir -p /app/src
WORKDIR /app
ADD . /app
RUN pip install -r requirements.txt
CMD ["python", "/app/src/index.py"]

EXPOSE 5001
