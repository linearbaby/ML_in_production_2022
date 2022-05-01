from project.data.utils import get_dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def process_numerical_features(data_row: pd.Series) -> pd.Series:
    si = SimpleImputer()
    data_row = si.fit_transform(data_row)
    ss = StandardScaler()
    data_row = ss.fit_transform(data_row)
    return data_row

def process_categorical_features(data_row: pd.Series) -> pd.Series:
    si = SimpleImputer(strategy='constant', fill_value='M')
    data_row = si.fit_transform(data_row)
    ohe = OneHotEncoder()
    data_row = ohe.fit_transform(data_row)
    return data_row


def extract_features():
