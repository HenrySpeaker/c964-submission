import dash
from dash import dcc, html, callback, no_update, Input, Output
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix

from app_utils import get_df_from_csv, get_onnx_prediction, load_stored_onnx_model, get_xy


# Registers the page with the Dash application
dash.register_page(__name__, path="/data-visualization")

# Defines the path to the .csv file containing data
CSV_PATH = "./current-data.csv"

# Converts the contents of the .csv to a Pandas DataFrame
df = get_df_from_csv(CSV_PATH)

# Extracts the names of columns for use choosing axes in the second scatterplot element
column_names = list(df.columns)[:-1]

# Creates a scatterplot with Ordinary Least Squares trendline comparing the height and weight differences across matches
height_weight_scatterplot = px.scatter(
    df,
    x="weight_diff",
    y="height_diff",
    trendline="ols",
    trendline_color_override="red",
    title="Scatterplot of Weight Difference Against Height Difference",
)

height_weight_scatterplot.update_layout(
    xaxis_title="Weight difference (kg)",
    yaxis_title="Height difference (cm)",
    title={"x": 0.5, "xanchor": "center"},
)

# Creates a correlation matrix across all pairs of features in the dataset
corr_matrix = df.corr()
corr_matrix_heatmap = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Correlation Matrix")
corr_matrix_heatmap.update_layout(
    title={"x": 0.5, "xanchor": "center"},
    yaxis_nticks=len(column_names),
    height=800,
    yaxis={"tickmode": "array", "ticktext": column_names},
)
corr_matrix_heatmap.update_traces(texttemplate="%{z:.3f}")

# Compares the frequencies of win streaks between higher-ranked rikishi and lower-ranked rikishi
win_streaks_histogram = px.histogram(
    df[["high_rikishi_win_streak", "low_rikishi_win_streak"]].rename(
        columns={
            "high_rikishi_win_streak": "Higher-ranked rikishi",
            "low_rikishi_win_streak": "Lower-ranked rikishi",
        }
    ),
    barmode="group",
    title="Frequencies of Win Streak for Higher and Lower Ranked Rikishi",
)
win_streaks_histogram.update_layout(
    xaxis_title="Win Streak",
    yaxis_title="Frequency",
    title={"x": 0.5, "xanchor": "center"},
    legend={"title": "Rikishi type"},
    xaxis={"tickmode": "linear", "tick0": 0, "dtick": 1},
)

# Creates a default scatterplot whose axes can be chosen by the user at a later point
scatterplot = px.scatter(df, "day", "height_diff", color="high_rikishi_won")


@callback(
    Output("scatterplot", "figure"),
    Output("scatterplot-err-msg", "children"),
    Input("h-axis-dropdown", "value"),
    Input("v-axis-dropdown", "value"),
)
def update_scatterplot(h_axis, v_axis):
    """Updates the axes of the scatterplot when the user chooses a new horizontal or vertical axis."""

    if h_axis == v_axis:
        return no_update, "Axis names must be different"

    if not h_axis or not v_axis:
        return no_update, "Both axes must be specified"

    return px.scatter(df, h_axis, v_axis, color="high_rikishi_won"), ""


# Fetches the most up-to-date ONNX model available, validates the model, and returns it to the application for use making predictions
model = load_stored_onnx_model()

# Creates a default CSS style for each data visualization element to visibly separate each element
default_style = {"border": "4px solid black", "border-radius": "4px", "margin-top": "16px"}

# Creates page layout with all data visualizations
layout = [
    html.Div(
        html.Div(
            dcc.Graph(figure=height_weight_scatterplot),
            style={
                "margin-left": "auto",
                "margin-right": "auto",
                "text-align": "center",
                "padding": "0",
            },
        ),
        style={
            "padding": "auto",
            "margin": "auto",
            "width": "80%",
            "border": "4px solid black",
            "border-radius": "4px",
        },
    ),
    dcc.Graph(figure=corr_matrix_heatmap, style={**default_style}),
    html.Div(
        [
            dcc.Graph(figure=win_streaks_histogram, style={**default_style, "width": "60%"}),
        ],
        style={"display": "flex", "flex-direction": "column", "align-items": "center"},
    ),
    html.Div(
        [
            html.Div(
                [
                    html.H3("Scatterplot Selection"),
                    html.P("Choose the field names for the horizontal and vertical axes of the scatterplot"),
                ],
                style={
                    "display": "flex",
                    "flex-direction": "column",
                    "justify-content": "center",
                    "align-items": "center",
                },
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.P("Horizontal axis:"),
                            dcc.Dropdown(column_names, "day", id="h-axis-dropdown", style={"width": "100%"}),
                        ],
                        style={"width": "30%"},
                    ),
                    html.Div(
                        [
                            html.P("Vertical axis:"),
                            dcc.Dropdown(column_names, "height_diff", id="v-axis-dropdown", style={"width": "100%"}),
                        ],
                        style={"width": "30%"},
                    ),
                ],
                style={
                    "display": "flex",
                    "flex-direction": "row",
                    "justify-content": "space-around",
                    "width": "60%",
                    "margin": "auto",
                },
            ),
            html.P(id="scatterplot-err-msg", style={"color": "red"}),
            dcc.Graph(figure=scatterplot, id="scatterplot"),
        ],
        style={**default_style, "padding": "10px"},
    ),
]


# Checks if model loaded successfully, or displays a message to the user if not
if model:

    def prediction_function(data):
        """Simple function to extract the boolean equivalent of the prediction from the ONNX model.
        Returns True if model predicts the higher-ranked rikishi has at least a 50% probability of winning match.
        Returns False otherwise."""

        return round(get_onnx_prediction(model, data)[0][0]) == 1

    # Fetches the input features as x_test and the real target values as y_test
    x_test, y_test = get_xy(CSV_PATH)

    # Converts x_test to a NumPy ndarray to allow the model to make predictions for each row of feature values
    x_test = x_test.to_numpy()

    # Generates a Pandas DataFrame containing the results predicted by the model
    y_pred = pd.DataFrame(np.squeeze(np.apply_along_axis(prediction_function, axis=1, arr=x_test)))

    # Generates a confusion matrix from the model's predictions and corrects the direction of each axis
    nn_conf_matrix = confusion_matrix(y_test, y_pred)
    nn_conf_matrix = np.flip(nn_conf_matrix, axis=(0, 1))

    # Creates a heatmap visualization of the confusion matrix
    nn_conf_matrix_heatmap = px.imshow(
        nn_conf_matrix,
        labels=dict(x="Predicted Value", y="Actual value"),
        x=["True", "False"],
        y=["True", "False"],
        title="Neural Network Confusion Matrix<br>(True = Higher-ranked rikishi won)",
        width=400,
    )
    nn_conf_matrix_heatmap.update_layout(
        title={"x": 0.5, "xanchor": "center"},
    )
    nn_conf_matrix_heatmap.update_traces(texttemplate="%{z:,.0d}")

    layout.append(dcc.Graph(figure=nn_conf_matrix_heatmap, style={**default_style}))

    # Generates a classification report from the predicted values and displays the results as a table
    cls_report = classification_report(y_test, y_pred, output_dict=True)
    layout.append(
        html.Div(
            [
                html.H3("Classification Report"),
                html.Table(
                    [
                        html.Thead(
                            [
                                html.Tr(
                                    [
                                        html.Td(""),
                                        html.Td("Precision"),
                                        html.Td("Recall"),
                                        html.Td("F1-score"),
                                        html.Td("Support"),
                                    ]
                                ),
                            ],
                            style={"font-size": "14px", "font-weight": "bold"},
                        ),
                        html.Tr(
                            [
                                html.Td("High rank won"),
                                html.Td(round(cls_report["1.0"]["precision"], 2)),
                                html.Td(round(cls_report["1.0"]["recall"], 2)),
                                html.Td(round(cls_report["1.0"]["f1-score"], 2)),
                                html.Td(cls_report["1.0"]["support"]),
                            ]
                        ),
                        html.Tr(
                            [
                                html.Td("High rank lost"),
                                html.Td(round(cls_report["0.0"]["precision"], 2)),
                                html.Td(round(cls_report["0.0"]["recall"], 2)),
                                html.Td(round(cls_report["0.0"]["f1-score"], 2)),
                                html.Td(cls_report["0.0"]["support"]),
                            ]
                        ),
                        html.Tr(
                            [
                                html.Td(id="td-row-line", style={"border-bottom": "1px solid black"}, colSpan=5),
                            ]
                        ),
                        html.Tr(
                            [
                                html.Td("Macro average"),
                                html.Td(round(cls_report["macro avg"]["precision"], 3)),
                                html.Td(round(cls_report["macro avg"]["recall"], 3)),
                                html.Td(round(cls_report["macro avg"]["f1-score"], 3)),
                                html.Td(round(cls_report["macro avg"]["support"], 3)),
                            ]
                        ),
                        html.Tr(
                            [
                                html.Td("Weighted average"),
                                html.Td(round(cls_report["weighted avg"]["precision"], 3)),
                                html.Td(round(cls_report["weighted avg"]["recall"], 3)),
                                html.Td(round(cls_report["weighted avg"]["f1-score"], 3)),
                                html.Td(round(cls_report["weighted avg"]["support"], 3)),
                            ]
                        ),
                        html.Tr(
                            [
                                html.Td(id="td-row-line", style={"border-bottom": "1px solid black"}, colSpan=5),
                            ]
                        ),
                        html.Tr(
                            [
                                html.Td("Accuracy"),
                                html.Td(),
                                html.Td(),
                                html.Td(round(cls_report["accuracy"], 3)),
                            ]
                        ),
                    ]
                ),
            ],
            style={"display": "flex", "flex-direction": "column", "align-items": "center", **default_style},
        ),
    )
else:
    layout.append(
        html.P("No models could be loaded, therefore no confusion matrix can be produced.", style={**default_style})
    )
