import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

# Daten laden
df = pd.read_csv("verbindungen_mit_kennzahlen.csv")
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
    ], style={'width': '45%', 'display': 'inline-block'}),

    html.Div([
        html.Label("Prognosemodell auswählen:"),
        dcc.Dropdown(
            id='modell-select',
            options=[
                {'label': 'Lineare Regression', 'value': 'LR'},
                {'label': 'Holt-Winters', 'value': 'HW'},
                {'label': 'ARIMA', 'value': 'ARIMA'},
                {'label': 'SARIMA', 'value': 'SARIMA'},
                {'label': 'Prophet', 'value': 'PROPHET'}
            ],
            value='LR'
        )
    ], style={'width': '45%', 'display': 'inline-block', 'marginLeft': '5%'}),

    dcc.Graph(id='zeitreihe'),

    html.H3("Auslastung & Statistiken"),
    html.Div(id='statistik-output'),

    html.H3("Prognose-Metriken"),
    html.Div(id='metriken-output')
])

@app.callback(
    Output('route-select', 'options'),
    Output('route-select', 'value'),
    Input('passagier-filter', 'value')
)
def filter_routen(min_passagiere):
    grouped = df.groupby(['ORIGIN', 'DEST'])
    routen = [{'label': 'Alle Flüge', 'value': 'ALL'}]

    for (origin, dest), gruppe in grouped:
        avg_pax = gruppe['PASSENGERS'].mean()
        if avg_pax >= min_passagiere:
            label = f"{origin} → {dest}"
            value = f"{origin}_{dest}"
            routen.append({'label': label, 'value': value})

    return routen, routen[0]['value']

@app.callback(
    Output('zeitreihe', 'figure'),
    Output('statistik-output', 'children'),
    Output('metriken-output', 'children'),
    Input('route-select', 'value'),
    Input('modell-select', 'value')
)
def update_dashboard(route, modell):
    if not route:
        return {}, "", ""

    if route == 'ALL':
        df_agg = df.groupby('DATE')['PASSENGERS'].sum().reset_index()
        df_agg['ORIGIN'] = 'ALL'
        df_agg['DEST'] = 'ALL'
        dff = df_agg
        title = 'Passagierzahlen: Alle Flüge'
    else:
        origin, dest = route.split("_")
        dff = df[(df['ORIGIN'] == origin) & (df['DEST'] == dest)].copy()
        title = f'Passagierzahlen: {origin} → {dest}'

    dff = dff.sort_values('DATE')

    all_dates = pd.date_range(start=dff['DATE'].min(), end=dff['DATE'].max(), freq='MS')
    dff = dff.set_index('DATE').reindex(all_dates).fillna(0.0).rename_axis('DATE').reset_index()

    # Zeitreihe
    fig = px.line(dff, x='DATE', y='PASSENGERS', title=title)

    stats_table = html.Table([
        html.Tr([html.Th("Metrik"), html.Th("Wert")]),
        html.Tr([html.Td("⌀ Passagiere"), html.Td(f"{dff['PASSENGERS'].mean():,.0f}")]),
        html.Tr([html.Td("Min/Max Passagiere"), html.Td(f"{dff['PASSENGERS'].min():,.0f} / {dff['PASSENGERS'].max():,.0f}")]),
        html.Tr([html.Td("Standardabweichung"), html.Td(f"{dff['PASSENGERS'].std():,.0f}")]),
    ])

    future_dates = pd.date_range(start='2024-01-01', end='2024-12-01', freq='MS')
    y_true_2024 = df[df['DATE'].between('2024-01-01', '2024-12-31')]
    if route == 'ALL':
        y_true_2024 = y_true_2024.groupby('DATE')['PASSENGERS'].sum()
    else:
        y_true_2024 = y_true_2024[(df['ORIGIN'] == origin) & (df['DEST'] == dest)].set_index('DATE')['PASSENGERS']

    forecast = []
    y_pred = []
    mae = rmse = r2 = None

    try:
        if modell == 'LR':
            dff['timestamp'] = (dff['DATE'] - dff['DATE'].min()).dt.days
            X = dff[['timestamp']]
            y = dff['PASSENGERS']
            model = LinearRegression().fit(X, y)
            future_X = (future_dates - dff['DATE'].min()).days.values.reshape(-1, 1)
            y_pred = model.predict(X)
            forecast = model.predict(future_X)

        elif modell == 'HW':
            model = ExponentialSmoothing(dff['PASSENGERS'], trend='add', seasonal='add', seasonal_periods=12)
            model_fit = model.fit()
            y_pred = model_fit.fittedvalues
            forecast = model_fit.forecast(12)

        elif modell == 'ARIMA':
            model = ARIMA(dff['PASSENGERS'], order=(1, 1, 1))
            model_fit = model.fit()
            y_pred = model_fit.predict(start=1, end=len(dff)-1, typ="levels")
            forecast = model_fit.forecast(12)

        elif modell == 'SARIMA':
            model = SARIMAX(dff['PASSENGERS'], order=(1,1,1), seasonal_order=(1,1,1,12))
            model_fit = model.fit(disp=False)
            y_pred = model_fit.fittedvalues
            forecast = model_fit.forecast(12)

        elif modell == 'PROPHET':
            prophet_df = dff[['DATE', 'PASSENGERS']].rename(columns={'DATE': 'ds', 'PASSENGERS': 'y'})
            model = Prophet()
            model.fit(prophet_df)
            future_df = pd.DataFrame({'ds': future_dates})
            forecast_df = model.predict(future_df)
            forecast = forecast_df['yhat'].values
            y_pred = model.predict(prophet_df)['yhat']

        # Prognose zeichnen
        fig.add_scatter(x=future_dates, y=forecast, name='Prognose 2024', mode='lines', line=dict(dash='dash'))

        # Metriken berechnen
        mae = mean_absolute_error(y_true_2024, forecast)
        rmse = np.sqrt(mean_squared_error(y_true_2024, forecast))
        r2 = r2_score(y_true_2024, forecast)

    except Exception as e:
        print("Fehler bei Prognose:", e)

    metriken_table = html.Table([
        html.Tr([html.Th("Metrik"), html.Th("Wert")]),
        html.Tr([html.Td("MAE"), html.Td(f"{mae:.2f}" if mae else "-")]),
        html.Tr([html.Td("RMSE"), html.Td(f"{rmse:.2f}" if rmse else "-")]),
        html.Tr([html.Td("R²"), html.Td(f"{r2:.2f}" if r2 else "-")]),
    ])

    return fig, stats_table, metriken_table

if __name__ == '__main__':
    app.run(debug=True)
