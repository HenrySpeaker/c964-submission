import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
from onnxruntime import InferenceSession
import pandas as pd


def get_df_from_csv(csv_file_path: str) -> pd.DataFrame:
    """Read data from csv and convert columns to numeric types"""

    def convert_value(val):
        return val == "True" or val == 1

    df = pd.read_csv(csv_file_path)
    results = df.iloc[:, -1]
    results = results.apply(convert_value)
    results = results.astype(int)
    df = df.apply(pd.to_numeric, errors="coerce")
    df.iloc[:, -1] = results
    return df


def get_xy(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = get_df_from_csv(csv_path)
    df = df.dropna()
    x = df.iloc[:, 0:-1]
    y = df.iloc[:, -1]
    return x, y


def load_onnx_model_from_file(file_path: str | Path) -> InferenceSession:
    model = onnx.load(file_path)
    onnx.checker.check_model(model)
    session = onnxruntime.InferenceSession(file_path, providers=["CPUExecutionProvider"])

    return session


def get_nn_models_by_ext(*args) -> list:
    dir_path = Path(".")
    dir_contents = os.listdir(dir_path)

    paths = [dir_path / name for name in dir_contents]

    valid_files = sorted((path for path in paths if path.suffix in args), key=lambda p: p.stat().st_mtime, reverse=True)

    return valid_files


def load_stored_onnx_model() -> InferenceSession | None:
    """Returns an ONNX model from the newest of the models stored in ./predictive_model/saved_models"""

    files = get_nn_models_by_ext(".onnx")
    model = None

    for model_file in files:
        try:
            model = load_onnx_model_from_file(model_file)

            # keep the first working model
            break
        except Exception:
            pass

    return model


def get_onnx_prediction(model: InferenceSession, model_input: list[float | int]) -> np.ndarray:
    return model.run(
        [model.get_outputs()[0].name],
        {model.get_inputs()[0].name: np.expand_dims(np.asarray(model_input).astype(dtype="float32"), axis=0)},
    )[0]
