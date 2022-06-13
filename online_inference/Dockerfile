FROM python:3.9

COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# frequently changing layers
COPY server.py server.py
COPY ./model_pickle/pipeline.pkl ./model_pickle/pipeline.pkl
COPY configs ./configs
COPY delayed_run.sh delayed_run.sh 

EXPOSE 8000
WORKDIR .
ENV MODEL_LOCATION=Google.disk

ENTRYPOINT [ "./delayed_run.sh" ]