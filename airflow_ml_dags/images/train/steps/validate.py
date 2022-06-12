import pandas as pd
from sklearn.metrics import accuracy_score
import click

import pickle


@click.command()
@click.option('--ds', envvar='DAY_STAMP', required=True)
def validate(ds: str):
    input_dir = f"/data/processed/{ds}/"
    x_test = pd.read_csv(input_dir + "data_test.csv")
    y_test = pd.read_csv(input_dir + "target_test.csv")
    X = pd.read_csv(input_dir + "data_train.csv")
    y = pd.read_csv(input_dir + "target_train.csv")

    model_dir = output_dir = f"/data/models/{ds}/"

    model = None
    with open(model_dir + "model.pkl", "rb") as f:
        model = pickle.load(f)

    # save metrics
    val_results = (
        f"SCORE ON TRAIN: {accuracy_score(y, model.predict(X))}\n"
        f"SCORE ON TEST: {accuracy_score(y_test, model.predict(x_test))}\n"
    )
    with open(output_dir + "metrics.txt", "w") as f:
        f.write(val_results)


if __name__ == "__main__":
    validate()
