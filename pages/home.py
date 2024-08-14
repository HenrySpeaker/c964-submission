import dash
from dash import dcc, html

dash.register_page(__name__, path="/")

intro_md = ""

with open("assets/sumo_intro.md", encoding="utf-8") as file:
    intro_md = file.read()

layout = [
    html.Div(
        [dcc.Markdown(intro_md, style={"width": "60%"})],
        style={"display": "flex", "flex-direction": "column", "align-items": "center"},
    )
]
