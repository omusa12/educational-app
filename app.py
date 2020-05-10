import os
import dash
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from utilities import prepare_ans_ques_ref, call_class
from tensorflow.keras.models import model_from_json

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

json_file = open(os.path.join("model", 'lstm_model.json'), 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(os.path.join("model", "best_model.h5"))

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "22rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE = {
    "margin-left": "22rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

navbar = html.Div(
    [
        html.H2("Alef EDUCATION", className="display-4"),
        html.Hr(),
        html.P(
            "Automated grading system for data science technical assessment", className="lead"
        ),
        html.Hr(),
        html.P(
            "Please input both question and answer, and for the reference answers fill up to 5 or less answers",
            className="lead"
        )
    ],
    style=SIDEBAR_STYLE,
)

result = html.Div(
    [
        html.H3('Result from form'),
        html.Div("Please Process", id="result")
    ]
)

controls = [
    dbc.Card(
        [
            dbc.FormGroup(
                [
                    dbc.Label("Test Question"),
                    dbc.Input(id="question", type="text"),
                ]
            ),
        ], body=True, style={"margin-bottom": "2rem"}
    ),
    dbc.Card(
        [
            dbc.FormGroup(
                [
                    dbc.Label("Test Answer"),
                    dbc.Input(id="answer", type="text"),
                ]
            ),
        ], body=True, style={"margin-bottom": "2rem"}
    ),
    dbc.Card(
        [
            dbc.FormGroup(
                [
                    dbc.Label("Test Reference Answers"),
                    dbc.Input(id="ref-ans-1", type="text", style={"margin-bottom": "1rem"}),
                    dbc.Input(id="ref-ans-2", type="text", style={"margin-bottom": "1rem"}),
                    dbc.Input(id="ref-ans-3", type="text", style={"margin-bottom": "1rem"}),
                    dbc.Input(id="ref-ans-4", type="text", style={"margin-bottom": "1rem"}),
                    dbc.Input(id="ref-ans-5", type="text")
                ]
            ),
        ], body=True, style={"margin-bottom": "3rem"}
    ),
    dbc.Button("Process", id="process", size="lg", className="mr-1")
]

body = dbc.Container([
    dbc.Row([
        html.H2('Testing Form:')
    ], style={"padding": "2rem 1rem"}),
    dbc.Row([
        dbc.Col(controls, md=10),
        dbc.Col(result, style={"margin-left": "2rem"})
    ])
], style=CONTENT_STYLE)


app.layout = html.Div([navbar, body])


@app.callback(Output('result', 'children'),
              [Input('process', "n_clicks")],
              [State("question", "value"),
               State("answer", "value"),
               State("ref-ans-1", "value"),
               State("ref-ans-2", "value"),
               State("ref-ans-3", "value"),
               State("ref-ans-4", "value"),
               State("ref-ans-5", "value")])
def make_prediction(n_clicks, question, answer, ref_ans_1, ref_ans_2, ref_ans_3, ref_ans_4, ref_ans_5):

    ref = [ref_ans_1, ref_ans_2, ref_ans_3, ref_ans_4, ref_ans_5]
    res = []
    for val in ref:
        if val is not None:
            res.append(val)
    cleaned_form = prepare_ans_ques_ref(answer, question, res)
    return call_class(model.predict(cleaned_form))


if __name__ == "__main__":
    app.run_server()
