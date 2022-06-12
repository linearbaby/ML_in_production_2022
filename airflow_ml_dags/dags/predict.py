from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount
from airflow.models import Variable


default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "predict_with_model",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(0, 2),
) as dag:
    predict = DockerOperator(
        image="airflow-predict",
        environment={
            'DAY_STAMP': "{{ds}}",
            'MODEL_PATH': str(Variable.get("model_path"))
        },
        network_mode="bridge",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="/home/artem/TECHNOPARK/ML_prod/airflow_ml_dags/data/", target="/data", type='bind')]
    )
