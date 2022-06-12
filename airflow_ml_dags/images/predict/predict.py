import pandas as pd

import os
import click
import pickle


@click.command()
@click.option('--ds', envvar='DAY_STAMP', required=True)
@click.option(
    '--model-path',
    envvar='MODEL_PATH',
    type=click.Path(exists=True),
    required=True
    )
def predict(ds: str, model_path):
    data_path = f"/data/raw/{ds}/"

    model = None
    with open(model_path + "model.pkl", "rb") as art_file:
        model = pickle.load(art_file)
    preprocess = None
    with open(model_path + "preprocess.pkl", "rb") as art_file:
        preprocess = pickle.load(art_file)

    data = pd.read_csv(data_path + "data.csv")
    print(data)
    data = pd.DataFrame(preprocess.transform(data))
    print(data)
    prediction = pd.DataFrame(model.predict(data))

    output_dir = f"/data/predictions/{ds}/"
    os.makedirs(output_dir, exist_ok=True)
    prediction.to_csv(output_dir + "predictions.csv", index=False)
    
    
if __name__ == "__main__":
    predict()
