import dash
from dash import Dash, dcc, html

# Initializes app, signals that additional pages can be found in the /pages directory
app = Dash(__name__, use_pages=True, suppress_callback_exceptions=True)

# Creates default layout for each page with <nav> element containing links at the top and individual page contents in the dash.page_container element
app.layout = html.Div(
    [
        html.Div(
            [
                html.H1("Sumo Predictor"),
                html.Nav(
                    [dcc.Link(f"{page['name']}", href=page["relative_path"]) for page in dash.page_registry.values()],
                    style={"border-bottom": "4px solid #343434", "margin": "20px 0", "padding-bottom": "20px"},
                ),
            ],
            style={"display": "flex", "flex-direction": "column", "align-items": "center"},
        ),
        dash.page_container,
    ],
)

# Runs the application when script is run
if __name__ == "__main__":
    app.run()
