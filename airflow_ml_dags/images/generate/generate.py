import click
import datetime
from sklearn.datasets import load_breast_cancer
import os


@click.command()
@click.option('--ds', envvar='DAY_STAMP', required=True)
@click.option('--from', 'from_source', envvar='GENERATE_FROM')
def generate(ds: str, from_source):
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    df = df.sample(frac=0.1, random_state=0)
    target = df.iloc[:, -1]
    df = df.iloc[:, :-1]

    output_dir = f"/data/raw/{ds}/"
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_dir + "data.csv", index=False)
    target.to_csv(output_dir + "target.csv", index=False)


if __name__ == "__main__":
    generate()
