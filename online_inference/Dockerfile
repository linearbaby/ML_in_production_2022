FROM python:3.9

COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# frequently changing layers
COPY server.py server.py
COPY ./model_pickle/pipeline.pkl ./model_pickle/pipeline.pkl
COPY configs ./configs

EXPOSE 8000
WORKDIR .
ENV MODEL_LOCATION=Google.disk

ENTRYPOINT [ "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000" ]