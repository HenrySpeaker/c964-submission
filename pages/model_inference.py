from fractions import Fraction

import dash
from dash import dcc, html, Input, Output, callback, no_update

from app_utils import load_stored_onnx_model, get_onnx_prediction


# Registers the page with the dash application
dash.register_page(__name__, path="/model-inference")

# Dictionary to contain store user inputs. Updated as the user changes input fields. Referenced when the user requests a prediction.
input_state = {
    "day": 1,
    "high_rikishi_height": 1,
    "low_rikishi_height": 1,
    "high_rikishi_weight": 1,
    "low_rikishi_weight": 1,
    "height_diff": 0,
    "weight_diff": 0,
    "high_rikishi_wins_over_low": 0,
    "high_rikishi_losses_to_low": 0,
    "high_rikishi_win_streak": 0,
    "low_rikishi_win_streak": 0,
    "high_rikishi_rank": 0,
    "low_rikishi_rank": 1,
    "high_rikishi_basho_wins": 0,
    "low_rikishi_basho_wins": 0,
    "high_rikishi_3_month_percentage": 0,
    "low_rikishi_3_month_percentage": 0,
    "high_rikishi_rank_change": 0,
    "low_rikishi_rank_change": 0,
    "rank_code_difference": 0,
    "high_rikishi_rank_code": 0,
    "low_rikishi_rank_code": 0,
    "high_rikishi_won": True,
}

# List of ranks to be used in the dropdown selector
rank_codes = ["Yokozuna", "Ozeki", "Sekiwake", "Komusubi", "Maegashira"]

# Set containing fields which cannot have a value of zero. Used to validate user input before prediction.
nonzero_fields = {
    "day",
    "high_rikishi_height",
    "low_rikishi_height",
    "high_rikishi_weight",
    "low_rikishi_weight",
    "low_rikishi_rank",
}

# Fetches the most up-to-date ONNX model available, validates the model, and returns it to the application for use making predictions
onnx_model = load_stored_onnx_model()


# The following functions update the input state based on the field being changed by the user
# Updates day of tournament
@callback(Input("day-dropdown", "value"))
def update_day(day):
    input_state["day"] = day


# Updates the rank code of the higher-ranked rikishi
@callback(Input("hr-rank-code", "value"))
def hr_rank_code(rank_code):
    input_state["high_rikishi_rank_code"] = int(rank_codes.index(rank_code))
    input_state["rank_code_difference"] = input_state["high_rikishi_rank_code"] - input_state["low_rikishi_rank_code"]


# Updates the rank code of the lower-ranked rikishi
@callback(Input("lr-rank-code", "value"))
def lr_rank_code(rank_code):
    input_state["low_rikishi_rank_code"] = int(rank_codes.index(rank_code))
    input_state["rank_code_difference"] = input_state["high_rikishi_rank_code"] - input_state["low_rikishi_rank_code"]


# Updates the height of the higher-ranked rikishi as well as the height difference between the two rikishi
@callback(Input("hr-height", "value"))
def hr_height(height):
    input_state["high_rikishi_height"] = height
    input_state["height_diff"] = input_state["high_rikishi_height"] - input_state["low_rikishi_height"]


# Updates the height of the lower-ranked rikishi as well as the height difference between the two rikishi
@callback(Input("lr-height", "value"))
def lr_height(height):
    input_state["low_rikishi_height"] = height
    input_state["height_diff"] = input_state["high_rikishi_height"] - input_state["low_rikishi_height"]


# Updates the weight of the higher-ranked rikishi as well as the weight difference between the two rikishi
@callback(Input("hr-weight", "value"))
def hr_weight(weight):
    input_state["high_rikishi_weight"] = weight
    input_state["weight_diff"] = input_state["high_rikishi_weight"] - input_state["low_rikishi_weight"]


# Updates the weight of the lower-ranked rikishi as well as the weight difference between the two rikishi
@callback(Input("lr-weight", "value"))
def lr_weight(weight):
    input_state["low_rikishi_weight"] = weight
    input_state["weight_diff"] = input_state["high_rikishi_weight"] - input_state["low_rikishi_weight"]


# Updates the number of career wins higher-ranked rikishi has over opponent
@callback(Input("hr-wins-opp", "value"))
def hr_wins_opp(wins):
    input_state["high_rikishi_wins_over_low"] = wins


# Updates the number of career wins lower-ranked rikishi has over opponent
@callback(Input("lr-wins-opp", "value"))
def lr_wins_opp(wins):
    input_state["high_rikishi_losses_to_low"] = wins


# Updates the current win streak of higher-ranked rikishi
@callback(Input("hr-win-streak", "value"))
def hr_win_streak(wins):
    input_state["high_rikishi_win_streak"] = wins


# Updates the current win streak of higher-ranked rikishi
@callback(Input("lr-win-streak", "value"))
def lr_win_streak(wins):
    input_state["low_rikishi_win_streak"] = wins


# Updates the rank of higher-ranked rikishi
@callback(Input("hr-rank", "value"))
def hr_rank(rank):
    input_state["high_rikishi_rank"] = rank


# Updates the rank of lower-ranked rikishi
@callback(Input("lr-rank", "value"))
def lr_rank(rank):
    input_state["low_rikishi_rank"] = rank


# Updates the win count in the current tournament for the higher-ranked rikishi
@callback(Input("hr-basho-wins", "value"))
def hr_basho_wins(wins):
    input_state["high_rikishi_basho_wins"] = wins


# Updates the win count in the current tournament for the lower-ranked rikishi
@callback(Input("lr-basho-wins", "value"))
def lr_basho_wins(wins):
    input_state["low_rikishi_basho_wins"] = wins


# Updates the 3-month win percentage of higher-ranked rikishi
@callback(Input("hr-win-pct", "value"))
def hr_win_pct(pct):
    input_state["high_rikishi_3_month_percentage"] = pct


# Updates the 3-month win percentage of lower-ranked rikishi
@callback(Input("lr-win-pct", "value"))
def lr_win_pct(pct):
    input_state["low_rikishi_3_month_percentage"] = pct


# Updates the rank change of the higher-ranked rikishi. Since ranks are counted from the top of the list, a change of +4 represents a demotion by 4 ranks.
@callback(Input("hr-rank-change", "value"))
def hr_rank_change(change):
    input_state["high_rikishi_rank_change"] = change


# Updates the rank change of the lower-ranked rikishi
@callback(Input("lr-rank-change", "value"))
def lr_rank_change(change):
    input_state["low_rikishi_rank_change"] = change


@callback(
    Output("out", "children"),
    Output("err", "children"),
    Output("pred-prob", "children"),
    Output("pred-odds", "children"),
    Input("predict", "n_clicks"),
)
def get_inference(_):
    """Validates the data provided by the user and makes a prediction when the Predict button is pressed.

    Displays the probability of the higher-ranked rikishi winning along with the equivalent odds.

    If inputs are not valid a message is displayed to the user describing the error."""

    # Boolean flag used to determine if the user has made a prediction yet. If the user hasn't, then no probabilities or odds are displayed.
    prediction_made = (
        max(
            input_state["high_rikishi_height"],
            input_state["high_rikishi_weight"],
            input_state["low_rikishi_height"],
            input_state["low_rikishi_weight"],
        )
        > 1
    )

    # Checks for any input equal to zero that cannot be used for prediction
    for k in input_state:
        if input_state[k] == 0 and k in nonzero_fields:
            return no_update, f"{k} field cannot be zero", "", ""

    # Checks to ensure that higher-ranked rikishi has a rank value lower than lower-ranked rikishi
    if input_state["rank_code_difference"] > 0 or input_state["high_rikishi_rank"] >= input_state["low_rikishi_rank"]:
        return no_update, "High rank must be at the same level or higher than low rank (i.e. lower rank value)", "", ""

    # Checks to ensure that win streaks are not exceeding the number of matches available
    if max(input_state["high_rikishi_win_streak"], input_state["low_rikishi_win_streak"]) >= input_state["day"]:
        return no_update, "Win streak cannot be greater than or equal to number of days in basho", "", ""

    # Checks that win totals are not exceeding the number of matches available
    if max(input_state["high_rikishi_basho_wins"], input_state["low_rikishi_basho_wins"]) >= input_state["day"]:
        return no_update, "Wins in current basho cannot be greater than or equal to number of days in basho", "", ""

    # Checks that win streaks do not exceed win totals
    if (
        input_state["high_rikishi_basho_wins"] < input_state["high_rikishi_win_streak"]
        or input_state["low_rikishi_basho_wins"] < input_state["low_rikishi_win_streak"]
    ):
        return no_update, "Wins in current basho cannot be less than win streak in basho", "", ""

    # Conversion of user input state to a list of only values the model needs for prediction
    model_input = [
        input_state["day"],
        input_state["height_diff"],
        input_state["weight_diff"],
        input_state["high_rikishi_wins_over_low"],
        input_state["high_rikishi_losses_to_low"],
        input_state["high_rikishi_win_streak"],
        input_state["low_rikishi_win_streak"],
        input_state["high_rikishi_rank"],
        input_state["low_rikishi_rank"],
        input_state["high_rikishi_basho_wins"],
        input_state["low_rikishi_basho_wins"],
        input_state["high_rikishi_3_month_percentage"],
        input_state["low_rikishi_3_month_percentage"],
        input_state["high_rikishi_rank_change"],
        input_state["low_rikishi_rank_change"],
        input_state["rank_code_difference"],
    ]

    probability = ""

    # Generates a prediction if possible
    try:
        probability = get_onnx_prediction(onnx_model, model_input)[0][0]
    except Exception as e:
        print(e)

    return (
        "",
        "",
        f"Probability of higher-ranked rikishi winning: {probability if prediction_made else '-'}",
        f"The odds of the higher-ranked rikishi winning are approximately: {get_odds_from_probability(probability.item() if prediction_made else -1)}",
    )


def get_odds_from_probability(probability: float) -> str:
    """Accepts a probability representing the likelihood of the higher-ranked rikishi winning.
    Converts the probability to odds which can be understood for betting purposes."""

    if probability == 1:
        return "Low-ranked rikishi has virtually no expectation to win"
    elif probability == 0:
        return "High-ranked rikishi has virtually no expectation to win"
    elif probability == -1:
        return "N/A"

    frac = Fraction(probability / (1 - probability)).limit_denominator(4)

    return f"{frac.numerator} to {frac.denominator}"


def build_rikishi_input_fields(rank, title):
    """Creates a <div> containing all of the input fields pertaining to each rikishi.
    Since two rikishi's data needs to be input using identical fields, this function is used to keep code DRY."""

    return html.Div(
        [
            html.H2(title),
            html.P("Enter rikishi's rank category:"),
            dcc.Dropdown(rank_codes, "Yokozuna", id=f"{rank}-rank-code", style={"width": "80%"}),
            html.P("Enter rikishi's height:"),
            dcc.Input("", id=f"{rank}-height", type="number", placeholder="Rikishi height", min=1, max=250),
            html.P("Enter rikishi's weight:"),
            dcc.Input("", id=f"{rank}-weight", type="number", placeholder="Rikishi weight", min=1, max=350),
            html.P("Enter rikishi's career wins over opponent:"),
            dcc.Input("", id=f"{rank}-wins-opp", type="number", placeholder="Career wins vs. opponent", min=0, max=100),
            html.P("Enter rikishi's current win streak:"),
            dcc.Input("", id=f"{rank}-win-streak", type="number", placeholder="Current win streak", min=0, max=15),
            html.P("Enter rikishi's rank (distance from top of banzuke):"),
            dcc.Input("", id=f"{rank}-rank", type="number", placeholder="Current rank", min=0, max=42),
            html.P("Enter rikishi's wins in current basho:"),
            dcc.Input("", id=f"{rank}-basho-wins", type="number", placeholder="Current basho wins", min=0, max=14),
            html.P("Enter rikishi's 3-month win percentage:"),
            dcc.Input(
                "",
                id=f"{rank}-win-pct",
                type="number",
                placeholder="Win percentage",
                min=0,
                max=1,
                step=0.001,
                style={"width": "40%"},
            ),
            html.P("Enter rikishi's rank change (from previous tournament to current):"),
            dcc.Input("", id=f"{rank}-rank-change", type="number", placeholder="Rank change", min=-42, max=42),
        ],
        id=rank,
        className="rikishi-input",
        style={"margin": "auto", "border": "4px solid #343434", "border-radius": "8px", "padding": "0 20px 16px"},
    )


# Creates layout of page, including input fields and model prediction output
layout = [
    html.Div(
        [
            html.Div(
                [
                    html.P("Day of tournament: "),
                    dcc.Dropdown([day for day in range(1, 16)], 1, id="day-dropdown", style={"width": "100%"}),
                ],
                style={"margin": "20px", "border": "4px solid", "padding": "10px", "border-radius": "8px"},
            ),
            html.Div(
                [
                    build_rikishi_input_fields("hr", "High-ranked rikishi"),
                    build_rikishi_input_fields("lr", "Low-ranked rikishi"),
                ],
                id="input-wrapper",
                style={"display": "flex"},
            ),
            html.Div(
                [
                    html.Button(
                        "Predict",
                        id="predict",
                        style={
                            "width": "100%",
                            "border": "4px solid #343434",
                            "font-size": "24px",
                            "margin": "20px auto",
                            "cursor": "pointer",
                        },
                    )
                ],
                style={"display": "flex", "flex-direction": "row", "align-content": "center"},
            ),
            html.Div(
                [
                    html.P(id="pred-prob", style={"font-size": "20px"}),
                    html.P(id="pred-odds", style={"font-size": "20px"}),
                ],
                id="prediction-results",
                style={"display": "flex", "flex-direction": "column", "align-items": "left"},
            ),
            html.P(id="err", style={"color": "red"}),
            html.P(id="out"),
        ],
        style={
            "display": "flex",
            "flex-direction": "column",
            "align-items": "center",
            "width": "100%",
        },
    ),
]
