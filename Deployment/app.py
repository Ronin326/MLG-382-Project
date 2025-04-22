import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import pandas as pd
import joblib

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# all features used in the trained model
feature_names = ['StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport', 'Engagement']

app = dash.Dash(__name__)
server = app.server  # For Render

app.layout = html.Div([
    html.H1("At-Risk Student Predictor", style={'textAlign': 'center'}),

    html.H2("Input student information", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Study Time Weekly (1 - 20)"),
        dcc.Input(id='study_time', type='number', value=5, min=1, max=20, step=1),
    ], style={'marginBottom': '10px'}),

    html.Div([
        html.Label("Absences (0 - 30)"),
        dcc.Input(id='absences', type='number', value=3, min=0, max=30, step=1),
    ], style={'marginBottom': '10px'}),

    html.Div([
        html.Label("Tutoring"),
        dcc.Dropdown(
            id='tutoring',
            options=[
                {'label': 'Yes', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=1,
            clearable=False
        ),
    ], style={'marginBottom': '10px'}),

    html.Div([
        html.Label("Parental Support (0–4)"),
        dcc.Slider(
            id='support',
            min=0,
            max=4,
            step=1,
            value=2,
            marks={i: str(i) for i in range(5)},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
    ], style={'marginBottom': '20px'}),

    html.Div([
        html.Label("Engagement (0–1)"),
        dcc.Slider(
            id='engagement',
            min=0,
            max=1,
            step=0.1,
            value=0.7,
            marks={i/10: str(i/10) for i in range(0, 11)},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
    ], style={'marginBottom': '20px'}),

    html.Div([
        html.Button('Predict', id='predict-btn', n_clicks=0),
    ], style={'textAlign': 'center', 'margin': '20px'}),

    html.Div(id='prediction-output', style={'fontWeight': 'bold', 'textAlign': 'center'})
], style={'width': '60%', 'margin': 'auto', 'padding': '20px'})

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('study_time', 'value'),
    State('absences', 'value'),
    State('tutoring', 'value'),
    State('support', 'value'),
    State('engagement', 'value'),
)
def predict_risk(n_clicks, study_time, absences, tutoring, support, engagement):
    if n_clicks == 0:
        return ""

    data = pd.DataFrame([[study_time, absences, tutoring, support, engagement]], columns=feature_names)  #using exported training files to predict
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0][1]

    result = "At Risk" if prediction == 1 else "Not At Risk"
    return f"Prediction: {result} (Risk Probability: {probability:.2%})"

if __name__ == '__main__':
    app.run_server(debug=True)
    