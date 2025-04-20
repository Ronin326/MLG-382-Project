import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import pandas as pd
import joblib
import gunicorn

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Define features expected by the model
feature_names = ['StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport', 'EngagementScore']

app = dash.Dash(__name__)
server = app.server  # For Render

app.layout = html.Div([
    html.H1("At-Risk Student Predictor"),

    html.Label("Study Time Weekly"),
    dcc.Input(id='study_time', type='number', value=5),

    html.Label("Absences"),
    dcc.Input(id='absences', type='number', value=3),

    html.Label("Tutoring (1=Yes, 0=No)"),
    dcc.Input(id='tutoring', type='number', value=1),

    html.Label("Parental Support (0–4)"),
    dcc.Input(id='support', type='number', value=2),

    html.Label("Engagement Score (0–1)"),
    dcc.Input(id='engagement', type='number', value=0.7),

    html.Br(),
    html.Button('Predict', id='predict-btn', n_clicks=0),
    html.Br(),
    html.Div(id='prediction-output')
])

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

    data = pd.DataFrame([[study_time, absences, tutoring, support, engagement]], columns=feature_names)
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0][1]

    result = "At Risk" if prediction == 1 else "Not At Risk"
    return f"Prediction: {result} (Risk Probability: {probability:.2%})"

if __name__ == '__main__':
    app.run_server(debug=True)