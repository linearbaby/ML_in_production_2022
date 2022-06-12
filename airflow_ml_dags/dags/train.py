from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "train_model",
    default_args=default_args,
    schedule_interval="@weekly",
    start_date=days_ago(0, 2),
) as dag:
    process_data = DockerOperator(
        image="airflow-process",
        environment={'DAY_STAMP': "{{ds}}"},
        network_mode="bridge",
        task_id="docker-airflow-process",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="/home/artem/TECHNOPARK/ML_prod/airflow_ml_dags/data/", target="/data", type='bind')]
    )

    split = DockerOperator(
        image="airflow-train",
        command="python split.py",
        environment={'DAY_STAMP': "{{ds}}"},
        network_mode="bridge",
        task_id="docker-airflow-split",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="/home/artem/TECHNOPARK/ML_prod/airflow_ml_dags/data/", target="/data", type='bind')]
    )

    train = DockerOperator(
        image="airflow-train",
        command="python train.py",
        environment={'DAY_STAMP': "{{ds}}", 'MODEL_TYPE': 'LogReg'},
        network_mode="bridge",
        task_id="docker-airflow-train",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="/home/artem/TECHNOPARK/ML_prod/airflow_ml_dags/data/", target="/data", type='bind')]
    )

    validate = DockerOperator(
        image="airflow-train",
        command="python validate.py",
        environment={'DAY_STAMP': "{{ds}}"},
        network_mode="bridge",
        task_id="docker-airflow-validate",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source="/home/artem/TECHNOPARK/ML_prod/airflow_ml_dags/data/", target="/data", type='bind')]
    )

    process_data >> split >> train >> validate
