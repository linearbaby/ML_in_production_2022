import numpy as np
import pandas as pd
import requests

if __name__ == "__main__":
    # загрузка датасета
    data = pd.read_csv('./data/datafiles/eval.csv')
    request_features = list(data.columns)

    # опрос сервера первыми 100 записями из датасета
    for i in range(100):
        request_data = data.iloc[i].tolist()
        print(len(request_data))
        response = requests.get(
            "http://0.0.0.0:8000/predict/",
            json={"data": [request_data], "features": request_features},
        )
        print(response.status_code)
        print(response.json())

    # проверка, что health возвращает 200
    print(
        'test /health, status code - '
        f'{requests.get("http://0.0.0.0:8000/health", json={}).status_code}')
