import pandas as pd
from sklearn.model_selection import train_test_split
import click


@click.command()
@click.option('--ds', envvar='DAY_STAMP', required=True)
def split(ds: str):
    output_dir = f"/data/processed/{ds}/"
    data = pd.read_csv(output_dir + "data.csv")
    target = pd.read_csv(output_dir + "target.csv")

    X, x_test, y, y_test = train_test_split(
        data,
        target,
        test_size=0.2,
        random_state=0)

    X.to_csv(output_dir + "data_train.csv", index=False)
    x_test.to_csv(output_dir + "data_test.csv", index=False)
    y.to_csv(output_dir + "target_train.csv", index=False)
    y_test.to_csv(output_dir + "target_test.csv", index=False)


if __name__ == "__main__":
    split()
