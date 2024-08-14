import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from app_utils import get_df_from_csv


def get_xy(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = get_df_from_csv(csv_path)
    df = df.dropna()
    x = df.iloc[:, 0:-1]
    y = df.iloc[:, -1]
    return x, y


def get_test_train_split(csv_path: str) -> list[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x, y = get_xy(csv_path)
    return train_test_split(x, y, train_size=0.8, random_state=1)


def get_test_train_cv_split(
    csv_path: str,
) -> list[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x, y = get_xy(csv_path)
    x, x_test, y, y_test = train_test_split(x, y, train_size=0.8, random_state=1)
    x, x_cv, y, y_cv = train_test_split(x, y, train_size=0.75, random_state=1)
    return x, x_cv, x_test, y, y_cv, y_test


def get_nn_models_by_ext(*args) -> list:
    dir_path = Path("./predictive_model/saved_models")
    dir_contents = os.listdir(dir_path)

    paths = [dir_path / name for name in dir_contents]

    valid_files = sorted((path for path in paths if path.suffix in args), key=lambda p: p.stat().st_mtime, reverse=True)

    return valid_files
