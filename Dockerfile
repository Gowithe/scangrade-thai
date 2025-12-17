FROM python:3.10.13-slim

WORKDIR /app
COPY . /app

RUN python -m pip install --upgrade pip \
 && python -m pip install --no-cache-dir -r requirements.txt

EXPOSE 10000
CMD ["gunicorn", "app:app", "--workers", "1", "--threads", "2", "--timeout", "180", "--bind", "0.0.0.0:10000"]
