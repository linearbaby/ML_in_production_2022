import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import click

import pickle
import os

model_configure_dict = {"LogReg": LogisticRegression, "GaussianNB": GaussianNB}


@click.command()
@click.option('--model-type', '-m',  envvar='MODEL_TYPE')
@click.option('--ds', envvar='DAY_STAMP', required=True)
def train(model_type: str, ds: str):
    input_dir = f"/data/processed/{ds}/"
    X = pd.read_csv(input_dir + "data_train.csv")
    y = pd.read_csv(input_dir + "target_train.csv").to_numpy().reshape((-1, ))

    if not model_type:
        model_type = "LogReg"
    model = model_configure_dict[model_type]()
    model.fit(X, y)

    # dump model
    output_dir = f"/data/models/{ds}/"
    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir + "model.pkl", "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    train()
