FROM python:3.10.12-slim

RUN mkdir startup_project

WORKDIR /startup_project

COPY . /startup_project/

RUN pip3 install --no-cache-dir --upgrade -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]