import copy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
import torch
import torch.nn as nn
import torch.optim as optim


def get_trained_sk_nn_model(x_train: pd.DataFrame, y_train: pd.DataFrame) -> MLPClassifier:
    """Train a scikit-learn neural network on the training data and return the model"""

    clf = MLPClassifier(
        solver="adam", alpha=1e-5, hidden_layer_sizes=(128, 64, 32, 16, 8), random_state=1, max_iter=500
    )
    clf.fit(x_train, y_train)
    return clf


def show_sk_nn_results(trained_model: MLPClassifier, x_test: pd.DataFrame, y_test: pd.DataFrame):
    """Display a variety of results describing the performance of the model against the provided testing data"""

    nn_predicted_outcomes = trained_model.predict(x_test)
    nn_predicted_probabilities = trained_model.predict_proba(x_test)
    nn_score = trained_model.score(x_test, y_test)

    print(nn_predicted_outcomes)
    print(nn_predicted_probabilities)
    print(nn_score)
    print(confusion_matrix(y_test, nn_predicted_outcomes))
    print(classification_report(y_test, nn_predicted_outcomes))


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        logits = self.stack(x)

        return self.sigmoid(logits)


def get_torch_x(x: pd.DataFrame) -> torch.Tensor:
    """Converts Pandas DataFrame of features to a Pytorch Tensor"""

    return torch.tensor(x.to_numpy(), dtype=torch.float32).cuda()
    # return torch.tensor(x, dtype=torch.float32).cuda()


def get_torch_y(y: pd.DataFrame) -> torch.Tensor:
    """Converts Pandas DataFrame of targets to a Pytorch Tensor"""

    return torch.tensor(y.to_numpy(), dtype=torch.float32).reshape(-1, 1).cuda()


def get_trained_pt_nn(
    x_train: pd.DataFrame,
    x_cv: pd.DataFrame,
    y_train: pd.DataFrame,
    y_cv: pd.DataFrame,
    save_path: str = "./predictive_model/saved_models/v0.pt",
) -> nn:
    """Train a pytorch neural network with the data provided and save the model's weights to a .pt file. Returns the trained model."""

    torch_train_x = get_torch_x(x_train)
    torch_train_y = get_torch_y(y_train)
    torch_test_x = get_torch_x(x_cv)
    torch_test_y = get_torch_y(y_cv)

    model = get_new_NN()

    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 500
    batch_size = 100
    batch_start = torch.arange(0, len(torch_train_x), batch_size)

    best_accuracy = -np.inf
    best_weights = None

    for epoch in range(num_epochs):
        print(f"current epoch: {epoch}")
        model.train()

        for start_idx in batch_start:
            features = torch_train_x[start_idx : start_idx + batch_size]
            targets = torch_train_y[start_idx : start_idx + batch_size]
            predicted_y = model(features)
            loss = loss_fn(predicted_y, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()

        with torch.no_grad():
            predicted_y = model(torch_test_x)

        curr_accuracy = get_accuracy(torch_test_y, predicted_y)
        curr_accuracy = float(curr_accuracy)

        if curr_accuracy > best_accuracy:
            print(f"new best accuracy: {curr_accuracy}")
            best_accuracy = curr_accuracy
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, save_path)

    model.load_state_dict(best_weights)
    torch.save(best_weights, save_path)

    return model


def get_accuracy(y_test: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Calculates the accuracy of the model's predictions against test data"""

    return (y_pred.round() == y_test).float().mean()


def get_new_NN():
    model = NN()
    device = "cuda"
    model = model.to(device)
    return model


def load_pt_model_from_file(file_path: str | Path) -> nn:
    """Returns a Pytorch model from the newest of the models stored in ./predictive_model/saved_models"""

    model = get_new_NN()
    weights = torch.load(file_path, weights_only=True)
    model.load_state_dict(weights)

    return model


def evaluate_pt_nn_accuracy(trained_model: nn, x_test: pd.DataFrame, y_test: pd.DataFrame):
    """Display a variety of results describing the performance of the model against the provided testing data"""

    torch_test_x = get_torch_x(x_test)
    torch_test_y = get_torch_y(y_test)

    with torch.no_grad():
        y_pred = trained_model(torch_test_x).round()

    accuracy = get_accuracy(torch_test_y, y_pred)
    print(f"Accuracy {accuracy}")
    print(confusion_matrix(y_pred.cpu(), torch_test_y.cpu()))
    print(classification_report(y_pred.cpu(), torch_test_y.cpu()))
