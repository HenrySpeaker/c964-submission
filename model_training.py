import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import torch
from torchinfo import summary

from pages.page_utils import load_stored_onnx_model
from predictive_model.model_utils import get_test_train_split, get_xy
from predictive_model.neural_networks import (
    get_trained_pt_nn,
    evaluate_pt_nn_accuracy,
    load_pt_model_from_file,
    get_torch_x,
)

# Define the path to the csv containing the training and testing data
CSV_PATH = "./data/storage/current-data.csv"

# Split the training and testing data
x_train, x_test, y_train, y_test = get_test_train_split(CSV_PATH)

# Define a model name
model_name = "v8-200001-202407-MJ"

# Train the model and print a summary describing its contents and efficacy
print("pytorch nn")
model = get_trained_pt_nn(x_train, x_test, y_train, y_test, f"./predictive_model/saved_models/{model_name}.pt")
summary(model, input_size=(1, 16))
evaluate_pt_nn_accuracy(model, x_test, y_test)

# Load the saved Pytorch model and verify it works properly
print(f"loading {model_name} from file")
v2 = load_pt_model_from_file(f"./predictive_model/saved_models/{model_name}.pt")
evaluate_pt_nn_accuracy(v2, x_test, y_test)
print(f"finished {model_name} pt evaluation")

# Export the Pytorch model to the ONNX format
torch_in = get_torch_x(x_test)
torch_out = v2(torch_in)
v2.train(False)
onnx_path = f"./predictive_model/saved_models/{model_name}.onnx"
print(f"exporting {model_name} to ONNX")
torch.onnx.export(
    v2,
    torch_in,
    onnx_path,
    export_params=True,
    opset_version=10,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},  # variable length axes
        "output": {0: "batch_size"},
    },
)

# Load the ONNX model and verify that it works as expected
x_test, y_test = get_xy(CSV_PATH)
print("loading stored ONNX model")
onnx_model = load_stored_onnx_model()
y_pred = onnx_model.run(
    [onnx_model.get_outputs()[0].name], {onnx_model.get_inputs()[0].name: x_test.to_numpy().astype(dtype="float32")}
)[0]
print(y_pred)
rounded = np.asarray([round(pred[0]) for pred in y_pred])
print(rounded)
print(y_test)
print((rounded == y_test.to_numpy()).mean())
print(confusion_matrix(rounded, y_test.to_numpy()))
print(classification_report(rounded, y_test.to_numpy(), output_dict=True))
print("-" * 20)
