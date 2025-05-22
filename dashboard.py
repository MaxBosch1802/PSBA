import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Daten laden
df = pd.read_csv("verbindungen_mit_kennzahlen.csv")

# Datum erzeugen
df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH']].assign(DAY=1))

# App initialisieren
app = dash.Dash(__name__)
app.title = "Flugauslastung Dashboard"

# Layout
app.layout = html.Div([
    html.H1("Flugstatistiken & Prognose Dashboard"),

    html.Div([
        html.Label("Mindest-Ø Passagiere pro Monat"),
        dcc.Slider(
            id='passagier-filter',
            min=0,
            max=30000,
            step=500,
            value=5000,
            marks={i: f'{i}' for i in range(0, 31000, 5000)},
            tooltip={"placement": "bottom", "always_visible": True}
        )
    ], style={'width': '60%', 'marginBottom': '30px'}),

    html.Div([
        html.Label("Wähle Route:"),
        dcc.Dropdown(id='route-select')
    ], style={'width': '50%', 'display': 'inline-block'}),

    dcc.Graph(id='zeitreihe'),

    html.H3("Auslastung & Statistiken"),
    html.Div(id='statistik-output'),

    html.H3("Prognose-Metriken"),
    html.Div(id='metriken-output')
])

# Callback zur Filterung der Routen
@app.callback(
    Output('route-select', 'options'),
    Output('route-select', 'value'),
    Input('passagier-filter', 'value')
)
def filter_routen(min_passagiere):
    grouped = df.groupby(['ORIGIN', 'DEST'])
    routen = []

    for (origin, dest), gruppe in grouped:
        avg_pax = gruppe['PASSENGERS'].mean()
        if avg_pax >= min_passagiere:
            label = f"{origin} → {dest}"
            value = f"{origin}_{dest}"
            routen.append({'label': label, 'value': value})

    if not routen:
        return [], None

    return routen, routen[0]['value']

# Dashboard aktualisieren
@app.callback(
    Output('zeitreihe', 'figure'),
    Output('statistik-output', 'children'),
    Output('metriken-output', 'children'),
    Input('route-select', 'value')
)
def update_dashboard(route):
    if not route:
        return {}, "", ""

    origin, dest = route.split("_")

    dff = df[(df['ORIGIN'] == origin) & (df['DEST'] == dest)].copy()
    dff = dff.sort_values('DATE')

    # Zeitreihe
    fig = px.line(dff, x='DATE', y='PASSENGERS', title=f'Passagierzahlen: {origin} → {dest}')

    # Statistiken
    stats_table = html.Table([
        html.Tr([html.Th("Metrik"), html.Th("Wert")]),
        html.Tr([html.Td("⌀ Passagiere"), html.Td(f"{dff['PASSENGERS'].mean():,.0f}")]),
        html.Tr([html.Td("⌀ Auslastung (%)"), html.Td(f"{dff['AUSLASTUNG'].mean() * 100:.2f}%")]),
        html.Tr([html.Td("Min/Max Passagiere"), html.Td(f"{dff['PASSENGERS'].min():,.0f} / {dff['PASSENGERS'].max():,.0f}")]),
        html.Tr([html.Td("Standardabweichung"), html.Td(f"{dff['PASSENGERS'].std():,.0f}")]),
    ])

    # Regressionsmodell
    dff['timestamp'] = (dff['DATE'] - dff['DATE'].min()).dt.days
    X = dff[['timestamp']]
    y = dff['PASSENGERS']

    model = LinearRegression()
    model.fit(X, y)

    # Prognose für 2024
    future_dates = pd.date_range(start='2024-01-01', end='2024-12-01', freq='MS')
    future_timestamps = (future_dates - dff['DATE'].min()).days.values.reshape(-1, 1)
    future_preds = model.predict(future_timestamps)

    fig.add_scatter(x=future_dates, y=future_preds, name='Prognose 2024', mode='lines', line=dict(dash='dash'))

    # Metriken
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    metriken_table = html.Table([
        html.Tr([html.Th("Metrik"), html.Th("Wert")]),
        html.Tr([html.Td("MAE"), html.Td(f"{mae:.2f}")]),
        html.Tr([html.Td("RMSE"), html.Td(f"{rmse:.2f}")]),
        html.Tr([html.Td("R²"), html.Td(f"{r2:.2f}")]),
    ])

    return fig, stats_table, metriken_table

# App starten
if __name__ == '__main__':
    app.run(debug=True)
