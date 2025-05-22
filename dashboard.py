import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Daten laden
df = pd.read_csv("verbindungen_mit_kennzahlen.csv")

# Datum aus YEAR & MONTH erzeugen
df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH']].assign(DAY=1))

# App initialisieren
app = dash.Dash(__name__)
app.title = "Flugauslastung Dashboard"

# Layout
app.layout = html.Div([
    html.H1("Flugstatistiken & Prognose Dashboard"),

    html.Div([
        html.Label("Wähle Route:"),
        dcc.Dropdown(
            id='route-select',
            options=[{'label': f"{o} → {d}", 'value': f"{o}_{d}"} for o, d in df[['ORIGIN', 'DEST']].drop_duplicates().values],
            value='CUN_BWI'
        ),
    ], style={'width': '30%', 'display': 'inline-block'}),

    dcc.Graph(id='zeitreihe'),

    html.H3("Auslastung & Statistiken"),
    html.Div(id='statistik-output'),

    html.H3("Prognose-Metriken"),
    html.Div(id='metriken-output')
])

# Callbacks
@app.callback(
    Output('zeitreihe', 'figure'),
    Output('statistik-output', 'children'),
    Output('metriken-output', 'children'),
    Input('route-select', 'value')
)
def update_dashboard(route):
    origin, dest = route.split("_")

    # Daten filtern
    dff = df[(df['ORIGIN'] == origin) & (df['DEST'] == dest)].copy()
    dff = dff.sort_values('DATE')

    # Zeitreihe
    fig = px.line(dff, x='DATE', y='PASSENGERS', title=f'Passagierzahlen: {origin} → {dest}')

    # Statistische Infos
    stats_table = html.Table([
        html.Tr([html.Th("Metrik"), html.Th("Wert")]),
        html.Tr([html.Td("⌀ Passagiere"), html.Td(f"{dff['PASSENGERS'].mean():,.0f}")]),
        html.Tr([html.Td("⌀ Auslastung (%)"), html.Td(f"{dff['AUSLASTUNG'].mean() * 100:.2f}%")]),
        html.Tr([html.Td("Min/Max Passagiere"), html.Td(f"{dff['PASSENGERS'].min():,.0f} / {dff['PASSENGERS'].max():,.0f}")]),
        html.Tr([html.Td("Standardabweichung"), html.Td(f"{dff['PASSENGERS'].std():,.0f}")]),
    ])

    # Prognose mit Linearer Regression (einfaches Beispiel)
    dff['timestamp'] = (dff['DATE'] - dff['DATE'].min()).dt.days
    X = dff[['timestamp']]
    y = dff['PASSENGERS']

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Prognosekurve hinzufügen
    dff['Prediction'] = y_pred
    fig.add_scatter(x=dff['DATE'], y=dff['Prediction'], name='Prognose', mode='lines')

    # Metriken
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
