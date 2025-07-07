import pandas as pd
import dash
from dash import dcc, html, Input, Output
from dash import dash_table
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Daten laden
df = pd.read_csv(
    "verbindungen_mit_kennzahlen.csv",
    dtype={"UNIQUE_CARRIER_ENTITY": str}
)
df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH']].assign(DAY=1))

# Mapping von Flughafencodes zu kompletten Namen laden
names_df = pd.read_csv(
    "gefilterte_verbindungen.csv",
    usecols=["ORIGIN", "ORIGIN_CITY_NAME", "DEST", "DEST_CITY_NAME"],
)
origin_names = (
    names_df[["ORIGIN", "ORIGIN_CITY_NAME"]]
    .drop_duplicates()
    .set_index("ORIGIN")["ORIGIN_CITY_NAME"]
    .to_dict()
)
dest_names = (
    names_df[["DEST", "DEST_CITY_NAME"]]
    .drop_duplicates()
    .set_index("DEST")["DEST_CITY_NAME"]
    .to_dict()
)

# Gewichtungen für verschiedene Szenarien des Routen-Rankings
scoring_scenarios = {
    'Wachstum': {
        'norm_mean_passagiere': 0.20,
        'norm_mean_load_factor': 0.10,
        'norm_forecast_growth': 0.50,
        'forecast_error_stability': 0.10,
        'norm_stabilitaet': 0.10,
    },
    'Effizienz': {
        'norm_mean_passagiere': 0.30,
        'norm_mean_load_factor': 0.40,
        'norm_forecast_growth': 0.00,
        'forecast_error_stability': 0.10,
        'norm_stabilitaet': 0.20,
    },
    'Risikoarm': {
        'norm_mean_passagiere': 0.20,
        'norm_mean_load_factor': 0.10,
        'norm_forecast_growth': 0.00,
        'forecast_error_stability': 0.40,
        'norm_stabilitaet': 0.30,
    },
    'Nachfrage': {
        'norm_mean_passagiere': 0.40,
        'norm_mean_load_factor': 0.30,
        'norm_forecast_growth': 0.10,
        'forecast_error_stability': 0.10,
        'norm_stabilitaet': 0.10,
    },
}


def full_airport(code: str, mapping: dict) -> str:
    """Hilfsfunktion für Darstellung 'CODE (Name)'"""
    name = mapping.get(code)
    return f"{code} ({name})" if name else code


df["ORIGIN_NAME"] = df["ORIGIN"].map(origin_names)
df["DEST_NAME"] = df["DEST"].map(dest_names)


def compute_route_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Berechne Metriken pro Route, die für alle Rankings benötigt werden.

    Um doppelte Berechnungen zu vermeiden, werden die Holt-Winters-Prognosen
    und Kennzahlen je Route nur einmal erstellt. Die spätere Gewichtung für
    die verschiedenen Szenarien greift auf dieses Ergebnis zurück.
    """

    hist = data[(data["DATE"].dt.year >= 2022) & (data["DATE"].dt.year <= 2023)]
    eval_2024 = data[data["DATE"].dt.year == 2024]

    rows = []

    for (origin, dest), grp_hist in hist.groupby(["ORIGIN", "DEST"]):
        grp_hist = grp_hist.sort_values("DATE")
        grp_eval = (
            eval_2024[(eval_2024["ORIGIN"] == origin) & (eval_2024["DEST"] == dest)]
            .sort_values("DATE")
        )

        ts = (
            grp_hist.groupby("DATE")["PASSENGERS"]
            .sum()
            .sort_index()
            .asfreq("MS", fill_value=0.0)
        )

        mean_passengers = grp_hist["PASSENGERS"].mean()
        mean_load_factor = grp_hist["AUSLASTUNG"].mean()
        std_passengers = grp_hist["PASSENGERS"].std()

        stability = 1 - (std_passengers / mean_passengers) if mean_passengers else 0.0

        if len(ts) > 1:
            try:
                model = ExponentialSmoothing(
                    ts,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=12,
                    damped_trend=True,
                    initialization_method="estimated",
                )
                model_fit = model.fit(optimized=True)
                forecast = model_fit.forecast(24)
            except Exception:
                forecast = np.repeat(mean_passengers, 24)
        else:
            forecast = np.repeat(mean_passengers, 24)

        forecast_2024 = forecast[:12]
        forecast_2025 = forecast[12:]

        if not grp_eval.empty:
            actual_2024 = grp_eval["PASSENGERS"].values[: len(forecast_2024)]
            rmse = np.sqrt(
                mean_squared_error(actual_2024, forecast_2024[: len(actual_2024)])
            )
        else:
            rmse = np.nan

        rrmse = rmse / mean_passengers if mean_passengers else np.nan

        mean_2023 = grp_hist[grp_hist["DATE"].dt.year == 2023]["PASSENGERS"].mean()
        forecast_growth = (
            forecast_2025.mean() - mean_2023 if not np.isnan(mean_2023) else 0.0
        )

        rows.append(
            {
                "ORIGIN": origin,
                "DEST": dest,
                "mean_passagiere": mean_passengers,
                "mean_load_factor": mean_load_factor,
                "forecast_growth": forecast_growth,
                "rrmse": rrmse,
                "stabilitaet": stability,
            }
        )

    return pd.DataFrame(rows)


def compute_route_ranking(metrics: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Gewichte vorhandene Metriken und erzeuge Ranking."""
    ranking = metrics.copy()
    if ranking.empty:
        return ranking

    # Fehlende RRMSE-Werte mit dem Spaltenmaximum auffüllen
    if ranking['rrmse'].isna().any():
        ranking['rrmse'] = ranking['rrmse'].fillna(ranking['rrmse'].max())

    for col in ['mean_passagiere', 'mean_load_factor', 'forecast_growth', 'rrmse', 'stabilitaet']:
        min_v = ranking[col].min()
        max_v = ranking[col].max()
        ranking[f'norm_{col}'] = (ranking[col] - min_v) / (max_v - min_v) if max_v != min_v else 0.0

    forecast_error_stability = 1 - ranking['norm_rrmse']

    ranking['score'] = (
        weights['norm_mean_passagiere'] * ranking['norm_mean_passagiere'] +
        weights['norm_mean_load_factor'] * ranking['norm_mean_load_factor'] +
        weights['norm_forecast_growth'] * ranking['norm_forecast_growth'] +
        weights['forecast_error_stability'] * forecast_error_stability +
        weights['norm_stabilitaet'] * ranking['norm_stabilitaet']
    ) * 100

    ranking['ampel'] = pd.cut(
        ranking['score'],
        bins=[-np.inf, 50, 75, np.inf],
        labels=['Nicht empfehlenswert', 'Beobachten', 'Empfehlung']
    )

    ranking = ranking.sort_values('score', ascending=False)
    ranking['score'] = ranking['score'].round(2)
    return ranking

# Metriken nur einmalig berechnen und anschließend je Szenario gewichten
metrics_df = compute_route_metrics(df)

precomputed_rankings = {
    scenario: compute_route_ranking(metrics_df, weights)
    for scenario, weights in scoring_scenarios.items()
}

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

    dcc.Tabs([
        dcc.Tab(label='Routen-Analyse', children=[
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

            html.Div(id='ranking-output', style={'marginTop': '20px'}),

            html.H3("Auslastung & Statistiken"),
            html.Div(id='statistik-output'),

            html.H3("Prognose-Metriken"),
            html.Div(id='metriken-output')
        ]),
        dcc.Tab(label='Routen-Ranking', children=[
            html.Div([
                html.Label("Szenario auswählen:"),
                dcc.Dropdown(
                    id='scenario-select',
                    options=[{'label': s, 'value': s} for s in scoring_scenarios.keys()],
                    value='Wachstum'
                )
            ], style={'width': '45%', 'marginBottom': '20px'}),
            dash_table.DataTable(
                id='ranking-table',
                columns=[
                    {'name': 'Origin', 'id': 'ORIGIN_FULL'},
                    {'name': 'Destination', 'id': 'DEST_FULL'},
                    {'name': 'Ø Passagiere', 'id': 'mean_passagiere', 'type': 'numeric', 'format': {'specifier': '.0f'}},
                    {'name': 'Ø Auslastung', 'id': 'mean_load_factor', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                    {'name': 'Stabilität', 'id': 'stabilitaet', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                    {'name': 'Prognosewachstum', 'id': 'forecast_growth', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                    {'name': 'RRMSE', 'id': 'rrmse', 'type': 'numeric', 'format': {'specifier': '.4f'}},
                    {'name': 'Score', 'id': 'score', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                    {'name': 'Empfehlung', 'id': 'ampel'}
                ],
                sort_action='native',
                page_size=20,
                data=[],
                style_cell={'textAlign': 'center'},
                style_data_conditional=[
                    {'if': {'filter_query': '{ampel} = "Empfehlung"', 'column_id': 'ampel'},
                     'backgroundColor': '#d4edda'},
                    {'if': {'filter_query': '{ampel} = "Beobachten"', 'column_id': 'ampel'},
                     'backgroundColor': '#fff3cd'},
                    {'if': {'filter_query': '{ampel} = "Nicht empfehlenswert"', 'column_id': 'ampel'},
                     'backgroundColor': '#f8d7da'}
                ]
            )
        ])
    ])
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
            label = f"{full_airport(origin, origin_names)} → {full_airport(dest, dest_names)}"
            value = f"{origin}_{dest}"
            routen.append({'label': label, 'value': value})

    return routen, routen[0]['value']


@app.callback(
    Output('ranking-table', 'data'),
    Input('passagier-filter', 'value'),
    Input('scenario-select', 'value')
)
def update_ranking_table(min_passagiere, scenario):
    ranking_df = precomputed_rankings.get(scenario, precomputed_rankings['Wachstum']).copy()
    ranking_df["ORIGIN_NAME"] = ranking_df["ORIGIN"].map(origin_names)
    ranking_df["DEST_NAME"] = ranking_df["DEST"].map(dest_names)
    ranking_df["ORIGIN_FULL"] = ranking_df["ORIGIN"].apply(lambda x: full_airport(x, origin_names))
    ranking_df["DEST_FULL"] = ranking_df["DEST"].apply(lambda x: full_airport(x, dest_names))
    filtered = ranking_df[ranking_df['mean_passagiere'] >= min_passagiere]
    return filtered.to_dict('records')

@app.callback(
    Output('zeitreihe', 'figure'),
    Output('statistik-output', 'children'),
    Output('metriken-output', 'children'),
    Output('ranking-output', 'children'),
    Input('route-select', 'value'),
    Input('modell-select', 'value'),
    Input('scenario-select', 'value')
)
def update_dashboard(route, modell, scenario):
    if not route:
        return {}, "", "", ""

    if route == 'ALL':
        df_agg = df.groupby('DATE').agg({'PASSENGERS': 'sum', 'SEATS': 'sum'}).reset_index()
        df_agg['AUSLASTUNG'] = df_agg['PASSENGERS'] / df_agg['SEATS']
        df_agg['ORIGIN'] = 'ALL'
        df_agg['DEST'] = 'ALL'
        dff = df_agg
        title = 'Passagierzahlen: Alle Flüge'
        ranking_div = html.Div("Kein Ranking für alle Flüge")
    else:
        origin, dest = route.split("_")
        dff = df[(df['ORIGIN'] == origin) & (df['DEST'] == dest)].copy()
        origin_full = full_airport(origin, origin_names)
        dest_full = full_airport(dest, dest_names)
        title = f'Passagierzahlen: {origin_full} → {dest_full}'
        ranking_df = precomputed_rankings.get(scenario, precomputed_rankings['Wachstum'])
        row = ranking_df[(ranking_df['ORIGIN'] == origin) & (ranking_df['DEST'] == dest)]
        if row.empty:
            ranking_div = html.Div("Keine Ranking-Daten verfügbar")
        else:
            score = row['score'].iloc[0]
            ampel = row['ampel'].iloc[0]
            color_map = {
                'Empfehlung': '#d4edda',
                'Beobachten': '#fff3cd',
                'Nicht empfehlenswert': '#f8d7da'
            }
            color = color_map.get(ampel, '#ffffff')
            ranking_div = html.Div([
                html.Strong(f"Score: {score:.2f}"),
                html.Span(f" – {ampel}", style={'marginLeft': '5px'})
            ], style={'padding': '10px', 'borderRadius': '5px', 'backgroundColor': color})

    dff = dff.sort_values('DATE')
    # Nur Daten von 2022 und 2023 verwenden
    dff = dff[(dff['DATE'].dt.year >= 2022) & (dff['DATE'].dt.year <= 2023)]


    all_dates = pd.date_range(start=dff['DATE'].min(), end=dff['DATE'].max(), freq='MS')
    dff = dff.set_index('DATE').reindex(all_dates).fillna(0.0).rename_axis('DATE').reset_index()

    # Zeitreihe
    fig = px.line(dff, x='DATE', y='PASSENGERS', title=title)

    stats_table = html.Table([
        html.Tr([html.Th("Metrik"), html.Th("Wert")]),
        html.Tr([html.Td("⌀ Passagiere"), html.Td(f"{dff['PASSENGERS'].mean():,.0f}")]),
        html.Tr([html.Td("⌀ Auslastung (%)"), html.Td(f"{dff['AUSLASTUNG'].mean() * 100:.2f}%")]),
        html.Tr([html.Td("Min/Max Passagiere"), html.Td(f"{dff['PASSENGERS'].min():,.0f} / {dff['PASSENGERS'].max():,.0f}")]),
        html.Tr([html.Td("Standardabweichung"), html.Td(f"{dff['PASSENGERS'].std():,.0f}")]),
    ])

    future_dates = pd.date_range(start='2024-01-01', end='2025-12-01', freq='MS')
    y_true_2024 = df[df['DATE'].between('2024-01-01', '2024-12-31')]
    if route == 'ALL':
        y_true_2024 = y_true_2024.groupby('DATE')['PASSENGERS'].sum()
    else:
        # Filter innerhalb des bereits eingegrenzten 2024-Datensatzes vornehmen
        y_true_2024 = y_true_2024[(y_true_2024['ORIGIN'] == origin) & (y_true_2024['DEST'] == dest)].set_index('DATE')['PASSENGERS']

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
            model = ExponentialSmoothing(
                dff['PASSENGERS'],
                trend='add',
                seasonal='add',
                seasonal_periods=12,
                damped_trend=True,
                initialization_method='estimated'
            )
            model_fit = model.fit(optimized=True)
            y_pred = model_fit.fittedvalues
            forecast = model_fit.forecast(24)

        elif modell == 'ARIMA':
            # Best parameters from cross validation
            model = ARIMA(dff['PASSENGERS'], order=(1, 0, 1))
            model_fit = model.fit()
            y_pred = model_fit.predict(start=1, end=len(dff)-1, typ="levels")
            forecast = model_fit.forecast(24)

        elif modell == 'SARIMA':
            model = SARIMAX(dff['PASSENGERS'], order=(0,0,0), seasonal_order=(1,1,0,12))
            model_fit = model.fit(disp=False)
            y_pred = model_fit.fittedvalues
            forecast = model_fit.forecast(24)

        elif modell == 'PROPHET':
            prophet_df = dff[['DATE', 'PASSENGERS']].rename(columns={'DATE': 'ds', 'PASSENGERS': 'y'})
            model = Prophet()
            model.fit(prophet_df)
            future_df = pd.DataFrame({'ds': future_dates})
            forecast_df = model.predict(future_df)
            forecast = forecast_df['yhat'].values
            y_pred = model.predict(prophet_df)['yhat']

        # Prognose zeichnen
        # Reale Daten für 2024 darstellen
        if not y_true_2024.empty:
            fig.add_scatter(
            x=y_true_2024.index,
            y=y_true_2024.values,
            name='Tatsächliche Daten 2024',
            mode='lines+markers',
            line=dict(color='green', width=2, dash='dot')
            )
        fig.add_scatter(
            x=future_dates,
            y=forecast,
            name='Prognose 2024-2025',
            mode='lines',
            line=dict(dash='dash')
        )

        # Metriken berechnen
        forecast_2024 = forecast[:len(y_true_2024)]
        mae = mean_absolute_error(y_true_2024, forecast_2024)
        rmse = np.sqrt(mean_squared_error(y_true_2024, forecast_2024))
        r2 = r2_score(y_true_2024, forecast_2024)

    except Exception as e:
        print("Fehler bei Prognose:", e)

    metriken_table = html.Table([
        html.Tr([html.Th("Metrik"), html.Th("Wert")]),
        html.Tr([html.Td("MAE"), html.Td(f"{mae:.2f}" if mae else "-")]),
        html.Tr([html.Td("RMSE"), html.Td(f"{rmse:.2f}" if rmse else "-")]),
        html.Tr([html.Td("R²"), html.Td(f"{r2:.2f}" if r2 else "-")]),
    ])

    return fig, stats_table, metriken_table, ranking_div

if __name__ == '__main__':
    app.run(debug=True)
