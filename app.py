import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import OrdinalEncoder


class FeatureDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        cols_to_drop = [col for col in self.columns_to_drop if col in X_copy.columns]
        return X_copy.drop(columns=cols_to_drop)


class FlightOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns_to_encode = ['class', 'stops', 'departure_time', 'arrival_time']
        self.category_orders = [
            ['Economy', 'Business'],
            ['zero', 'one', 'two_or_more'],
            ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night'],
            ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night'],
        ]
        self.encoder = OrdinalEncoder(
            categories=self.category_orders,
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )

    def fit(self, X, y=None):
        self.encoder.fit(X[self.columns_to_encode])
        self.is_fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self, 'is_fitted_')
        X_copy = X.copy()
        X_copy[self.columns_to_encode] = self.encoder.transform(X_copy[self.columns_to_encode])
        return X_copy


class Log1pTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_log):
        self.columns_to_log = columns_to_log

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns_to_log:
            if col in X_copy.columns:
                X_copy[col] = np.log1p(X_copy[col])
        return X_copy


df = pd.read_csv("Clean_Dataset.csv")

import __main__
__main__.FeatureDropper = FeatureDropper
__main__.FlightOrdinalEncoder = FlightOrdinalEncoder
__main__.Log1pTransformer = Log1pTransformer
model = joblib.load("flight_price_model.pkl")

AIRLINES      = sorted(df["airline"].unique())
CITIES        = sorted(df["source_city"].unique())
TIME_SLOTS    = ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]
STOPS_OPTIONS = ["zero", "one", "two_or_more"]
CLASSES       = ["Economy", "Business"]

COLORS = {
    "bg":      "#F8F9FA",
    "card":    "#FFFFFF",
    "border":  "#DEE2E6",
    "accent":  "#2563EB",
    "accent2": "#DC2626",
    "accent3": "#16A34A",
    "text":    "#111827",
    "subtext": "#6B7280",
    "muted":   "#F3F4F6",
}

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], family="'Inter', sans-serif"),
        xaxis=dict(gridcolor="#E5E7EB", linecolor="#E5E7EB", zerolinecolor="#E5E7EB"),
        yaxis=dict(gridcolor="#E5E7EB", linecolor="#E5E7EB", zerolinecolor="#E5E7EB"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#E5E7EB"),
        colorway=[COLORS["accent"], COLORS["accent2"], COLORS["accent3"], "#7C3AED", "#D97706", "#0891B2"],
    )
)

avg_price     = int(df["price"].mean())
min_price     = int(df["price"].min())
max_price     = int(df["price"].max())
total_records = f"{len(df):,}"


def card(children, style=None):
    base = {
        "background": COLORS["card"],
        "border": f"1px solid {COLORS['border']}",
        "borderRadius": "10px",
        "padding": "24px",
        "marginBottom": "20px",
        "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
    }
    if style:
        base.update(style)
    return html.Div(children, style=base)


def question_badge(letter, question_text):
    return html.Div(style={"display": "flex", "alignItems": "flex-start", "gap": "12px", "marginBottom": "16px"}, children=[
        html.Div(f"Q{letter}", style={
            "background": COLORS["accent"],
            "color": "white",
            "borderRadius": "6px",
            "padding": "2px 10px",
            "fontSize": "12px",
            "fontWeight": "700",
            "flexShrink": "0",
            "marginTop": "1px",
        }),
        html.Span(question_text, style={
            "fontSize": "13px",
            "color": COLORS["subtext"],
            "lineHeight": "1.5",
            "fontStyle": "italic",
        }),
    ])


def field_label(text):
    return html.Label(text, style={
        "color": COLORS["subtext"],
        "fontSize": "11px",
        "fontWeight": "600",
        "letterSpacing": "0.06em",
        "textTransform": "uppercase",
        "marginBottom": "6px",
        "display": "block",
    })


def dropdown(id_, options, value):
    return dcc.Dropdown(
        id=id_,
        options=[{"label": o, "value": o} for o in options],
        value=value,
        clearable=False,
        style={"fontSize": "13px"},
    )


def tip(text, color):
    return html.Div(style={"display": "flex", "alignItems": "flex-start", "gap": "10px"}, children=[
        html.Span(">", style={"color": color, "flexShrink": "0", "fontWeight": "700", "fontSize": "16px"}),
        html.Span(text, style={"color": COLORS["text"], "fontSize": "13px", "lineHeight": "1.6"}),
    ])


app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.index_string = '''
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>FlightIQ - India Flight Price Dashboard</title>
    {%favicon%}
    {%css%}
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #F8F9FA; color: #111827; font-family: 'Inter', sans-serif; }

        .tab {
            background: #FFFFFF !important;
            border: 1px solid #DEE2E6 !important;
            color: #6B7280 !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 13px !important;
            font-weight: 500 !important;
            padding: 10px 24px !important;
            border-radius: 8px 8px 0 0 !important;
        }
        .tab--selected {
            background: #2563EB !important;
            color: #FFFFFF !important;
            border-color: #2563EB !important;
            font-weight: 600 !important;
        }

        #predict-btn {
            background: #2563EB;
            color: white;
            border: none;
            padding: 13px 28px;
            border-radius: 8px;
            font-family: 'Inter', sans-serif;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
            margin-top: 8px;
            letter-spacing: 0.03em;
            transition: background 0.2s;
        }
        #predict-btn:hover { background: #1D4ED8; }

        .stat-card {
            background: #FFFFFF;
            border: 1px solid #DEE2E6;
            border-radius: 10px;
            padding: 20px 24px;
            flex: 1;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .stat-number {
            font-family: 'Syne', sans-serif;
            font-size: 26px;
            font-weight: 800;
            line-height: 1.2;
        }
        .stat-label {
            color: #6B7280;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-top: 4px;
        }

        .rc-slider-track { background-color: #2563EB !important; }
        .rc-slider-handle { border-color: #2563EB !important; background: #2563EB !important; }
        .rc-slider-rail { background-color: #E5E7EB !important; }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
'''

app.layout = html.Div(style={"minHeight": "100vh", "background": COLORS["bg"]}, children=[

    html.Div(style={
        "background": COLORS["card"],
        "borderBottom": f"1px solid {COLORS['border']}",
        "padding": "18px 40px",
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "space-between",
        "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
    }, children=[
        html.Div([
            html.Span("FlightIQ", style={"fontFamily": "'Syne', sans-serif", "fontSize": "24px", "fontWeight": "800", "color": COLORS["accent"]}),
            html.Span("  |  India Domestic Flight Price Analysis", style={"color": COLORS["subtext"], "fontSize": "13px"}),
        ]),
        html.Div(style={"display": "flex", "gap": "8px"}, children=[
            html.Div(f"Q{l}", style={
                "background": COLORS["muted"], "color": COLORS["accent"],
                "border": f"1px solid {COLORS['border']}",
                "borderRadius": "5px", "padding": "2px 9px",
                "fontSize": "11px", "fontWeight": "700",
            }) for l in ["A", "B", "C", "D", "E"]
        ]),
    ]),

    html.Div(style={"display": "flex", "gap": "16px", "padding": "24px 40px 0"}, children=[
        html.Div([html.Div(total_records,       className="stat-number", style={"color": COLORS["accent"]}),  html.Div("Total Flights",   className="stat-label")], className="stat-card"),
        html.Div([html.Div(f"Rs.{avg_price:,}", className="stat-number", style={"color": COLORS["accent3"]}), html.Div("Avg Price (INR)", className="stat-label")], className="stat-card"),
        html.Div([html.Div(f"Rs.{min_price:,}", className="stat-number", style={"color": COLORS["subtext"]}), html.Div("Lowest Price",    className="stat-label")], className="stat-card"),
        html.Div([html.Div(f"Rs.{max_price:,}", className="stat-number", style={"color": COLORS["accent2"]}), html.Div("Highest Price",   className="stat-label")], className="stat-card"),
        html.Div([html.Div("6",                 className="stat-number", style={"color": "#7C3AED"}),          html.Div("Airlines",        className="stat-label")], className="stat-card"),
    ]),

    html.Div(style={"padding": "24px 40px"}, children=[
        dcc.Tabs(value="insights", children=[

            dcc.Tab(label="Research Questions", value="insights", className="tab", selected_className="tab--selected", children=[
                html.Div(style={"marginTop": "20px"}, children=[

                    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"}, children=[

                        card([
                            question_badge("A", "Does price vary with Airlines?"),
                            html.H3("Average & Spread of Price by Airline", style={"fontSize": "15px", "fontWeight": "600", "marginBottom": "4px"}),
                            html.P("Box plot shows the full price distribution per airline — median, range, and outliers.", style={"color": COLORS["subtext"], "fontSize": "12px", "marginBottom": "14px"}),
                            dcc.Graph(id="qa-airline-chart", config={"displayModeBar": False}),
                        ]),

                        card([
                            question_badge("E", "How does the ticket price vary between Economy and Business class?"),
                            html.H3("Economy vs Business Class Pricing", style={"fontSize": "15px", "fontWeight": "600", "marginBottom": "4px"}),
                            html.P("Violin plot shows the full price density and spread for each class.", style={"color": COLORS["subtext"], "fontSize": "12px", "marginBottom": "14px"}),
                            dcc.Graph(id="qe-class-chart", config={"displayModeBar": False}),
                        ]),

                    ]),

                    card([
                        question_badge("B", "How is the price affected when tickets are bought in just 1 or 2 days before departure?"),
                        html.H3("Price vs Days Left Before Departure", style={"fontSize": "15px", "fontWeight": "600", "marginBottom": "4px"}),
                        html.P("The line shows how average ticket price rises sharply as the departure date approaches. The red zone highlights the last 1-2 days penalty.", style={"color": COLORS["subtext"], "fontSize": "12px", "marginBottom": "14px"}),
                        dcc.Graph(id="qb-days-chart", config={"displayModeBar": False}),
                    ]),

                    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"}, children=[

                        card([
                            question_badge("C", "Does ticket price change based on the departure time and arrival time?"),
                            html.H3("Price by Departure Time", style={"fontSize": "15px", "fontWeight": "600", "marginBottom": "4px"}),
                            html.P("Average price across time slots for Economy and Business separately.", style={"color": COLORS["subtext"], "fontSize": "12px", "marginBottom": "14px"}),
                            dcc.Graph(id="qc-dep-chart", config={"displayModeBar": False}),
                        ]),

                        card([
                            question_badge("C", "Does ticket price change based on the departure time and arrival time?"),
                            html.H3("Price by Arrival Time", style={"fontSize": "15px", "fontWeight": "600", "marginBottom": "4px"}),
                            html.P("Average price across arrival time slots for Economy and Business separately.", style={"color": COLORS["subtext"], "fontSize": "12px", "marginBottom": "14px"}),
                            dcc.Graph(id="qc-arr-chart", config={"displayModeBar": False}),
                        ]),

                    ]),

                    card([
                        question_badge("D", "How the price changes with change in Source and Destination?"),
                        html.H3("Average Ticket Price by Route (Source to Destination)", style={"fontSize": "15px", "fontWeight": "600", "marginBottom": "4px"}),
                        html.P("Each cell shows the average price for that route. Darker blue = cheaper, darker red = more expensive.", style={"color": COLORS["subtext"], "fontSize": "12px", "marginBottom": "14px"}),
                        dcc.Graph(id="qd-route-chart", config={"displayModeBar": False}),
                    ]),

                ]),
            ]),

            dcc.Tab(label="Price Predictor", value="predictor", className="tab", selected_className="tab--selected", children=[
                html.Div(style={"marginTop": "20px", "display": "grid", "gridTemplateColumns": "380px 1fr", "gap": "24px"}, children=[

                    card([
                        html.H3("Flight Details", style={"fontSize": "16px", "fontWeight": "700", "marginBottom": "20px", "borderBottom": f"1px solid {COLORS['border']}", "paddingBottom": "12px"}),

                        field_label("Airline"),
                        dropdown("pred-airline", AIRLINES, AIRLINES[0]),
                        html.Div(style={"height": "14px"}),

                        field_label("Seat Class"),
                        dropdown("pred-class", CLASSES, "Economy"),
                        html.Div(style={"height": "14px"}),

                        field_label("Source City"),
                        dropdown("pred-source", CITIES, "Delhi"),
                        html.Div(style={"height": "14px"}),

                        field_label("Destination City"),
                        dropdown("pred-dest", CITIES, "Mumbai"),
                        html.Div(style={"height": "14px"}),

                        field_label("Departure Time"),
                        dropdown("pred-dep-time", TIME_SLOTS, "Morning"),
                        html.Div(style={"height": "14px"}),

                        field_label("Arrival Time"),
                        dropdown("pred-arr-time", TIME_SLOTS, "Afternoon"),
                        html.Div(style={"height": "14px"}),

                        field_label("Number of Stops"),
                        dropdown("pred-stops", STOPS_OPTIONS, "zero"),
                        html.Div(style={"height": "20px"}),

                        field_label("Flight Duration (hours)"),
                        dcc.Slider(id="pred-duration", min=0.5, max=50, step=0.5, value=2.5,
                                   marks={i: str(i) for i in [1, 10, 20, 30, 40, 50]},
                                   tooltip={"placement": "bottom", "always_visible": True}),
                        html.Div(style={"height": "20px"}),

                        field_label("Days Left Until Flight"),
                        dcc.Slider(id="pred-days", min=1, max=49, step=1, value=30,
                                   marks={i: str(i) for i in [1, 10, 20, 30, 40, 49]},
                                   tooltip={"placement": "bottom", "always_visible": True}),
                        html.Div(style={"height": "24px"}),

                        html.Button("PREDICT PRICE", id="predict-btn", n_clicks=0),

                    ], style={"height": "fit-content"}),

                    html.Div([

                        html.Div(id="prediction-result", children=[
                            card([html.Div("Select your flight details and click Predict", style={"color": COLORS["subtext"], "textAlign": "center", "padding": "40px", "fontSize": "14px"})]),
                        ]),

                        card([
                            html.H3("What Affects Your Price?", style={"fontSize": "15px", "fontWeight": "600", "marginBottom": "16px"}),
                            dcc.Graph(id="feature-importance-chart", config={"displayModeBar": False}),
                        ]),

                        card([
                            html.H3("Key Findings", style={"fontSize": "15px", "fontWeight": "600", "marginBottom": "16px"}),
                            html.Div(style={"display": "flex", "flexDirection": "column", "gap": "10px"}, children=[
                                tip("(A) Vistara and Air India are significantly more expensive than budget carriers", COLORS["accent"]),
                                tip("(B) Booking 1-2 days before departure can cost 2-3x more than booking early", COLORS["accent2"]),
                                tip("(C) Early Morning and Late Night flights are typically the cheapest time slots", COLORS["accent3"]),
                                tip("(D) Routes between metro hubs like Delhi-Mumbai tend to be the most competitive", "#7C3AED"),
                                tip("(E) Business class tickets average 5-8x higher than Economy on the same route", "#D97706"),
                            ]),
                        ]),
                    ]),
                ]),
            ]),

        ]),
    ]),
])


@app.callback(Output("qa-airline-chart", "figure"), Input("qa-airline-chart", "id"))
def qa_airline(_):
    order = df.groupby("airline")["price"].median().sort_values(ascending=False).index.tolist()
    fig = px.box(df, x="airline", y="price", color="airline",
                 category_orders={"airline": order},
                 labels={"price": "Price (INR)", "airline": "Airline"})
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=320, showlegend=False)
    fig.update_traces(hovertemplate="<b>%{x}</b><br>Rs.%{y:,.0f}<extra></extra>")
    return fig


@app.callback(Output("qe-class-chart", "figure"), Input("qe-class-chart", "id"))
def qe_class(_):
    fig = px.violin(df, x="class", y="price", color="class", box=True,
                    color_discrete_map={"Economy": COLORS["accent"], "Business": COLORS["accent2"]},
                    labels={"price": "Price (INR)", "class": "Seat Class"})
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=320, showlegend=False)
    fig.update_traces(hovertemplate="<b>%{x}</b><br>Rs.%{y:,.0f}<extra></extra>")
    return fig


@app.callback(Output("qb-days-chart", "figure"), Input("qb-days-chart", "id"))
def qb_days(_):
    avg = df.groupby(["days_left", "class"])["price"].mean().reset_index()
    fig = px.line(avg, x="days_left", y="price", color="class",
                  color_discrete_map={"Economy": COLORS["accent"], "Business": COLORS["accent2"]},
                  labels={"price": "Avg Price (INR)", "days_left": "Days Left Before Departure", "class": "Class"})
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=300)
    fig.add_vrect(x0=1, x1=2, fillcolor=COLORS["accent2"], opacity=0.10,
                  annotation_text="1-2 Days Before", annotation_position="top right",
                  annotation_font_color=COLORS["accent2"])
    fig.add_vrect(x0=15, x1=49, fillcolor=COLORS["accent3"], opacity=0.06,
                  annotation_text="Best Booking Window", annotation_position="top left",
                  annotation_font_color=COLORS["accent3"])
    return fig


@app.callback(Output("qc-dep-chart", "figure"), Input("qc-dep-chart", "id"))
def qc_dep(_):
    time_order = ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]
    avg = df.groupby(["departure_time", "class"])["price"].mean().reset_index()
    avg["departure_time"] = pd.Categorical(avg["departure_time"], categories=time_order, ordered=True)
    avg = avg.sort_values("departure_time")
    fig = px.bar(avg, x="departure_time", y="price", color="class", barmode="group",
                 color_discrete_map={"Economy": COLORS["accent"], "Business": COLORS["accent2"]},
                 labels={"price": "Avg Price (INR)", "departure_time": "Departure Time", "class": "Class"})
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=300)
    fig.update_traces(hovertemplate="<b>%{x}</b> - %{data.name}<br>Rs.%{y:,.0f}<extra></extra>")
    return fig


@app.callback(Output("qc-arr-chart", "figure"), Input("qc-arr-chart", "id"))
def qc_arr(_):
    time_order = ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]
    avg = df.groupby(["arrival_time", "class"])["price"].mean().reset_index()
    avg["arrival_time"] = pd.Categorical(avg["arrival_time"], categories=time_order, ordered=True)
    avg = avg.sort_values("arrival_time")
    fig = px.bar(avg, x="arrival_time", y="price", color="class", barmode="group",
                 color_discrete_map={"Economy": COLORS["accent"], "Business": COLORS["accent2"]},
                 labels={"price": "Avg Price (INR)", "arrival_time": "Arrival Time", "class": "Class"})
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=300)
    fig.update_traces(hovertemplate="<b>%{x}</b> - %{data.name}<br>Rs.%{y:,.0f}<extra></extra>")
    return fig


@app.callback(Output("qd-route-chart", "figure"), Input("qd-route-chart", "id"))
def qd_route(_):
    pivot = df.groupby(["source_city", "destination_city"])["price"].mean().unstack()
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[[0, "#DBEAFE"], [0.5, "#2563EB"], [1, "#DC2626"]],
        hoverongaps=False,
        hovertemplate="From: <b>%{y}</b><br>To: <b>%{x}</b><br>Avg Price: Rs.%{z:,.0f}<extra></extra>",
        text=[[f"Rs.{v:,.0f}" if not np.isnan(v) else "" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont={"size": 11, "color": "white"},
    ))
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=340)
    return fig


@app.callback(Output("feature-importance-chart", "figure"), Input("feature-importance-chart", "id"))
def feature_importance(_):
    features   = ["Seat Class", "Days Left", "Duration", "Airline", "Source City", "Destination City", "Stops", "Dep. Time", "Arr. Time"]
    importance = [0.42, 0.22, 0.14, 0.09, 0.04, 0.04, 0.03, 0.01, 0.01]
    colors     = [COLORS["accent2"] if i == 0 else COLORS["accent"] for i in range(len(features))]
    fig = go.Figure(go.Bar(x=importance, y=features, orientation="h", marker_color=colors,
                           hovertemplate="<b>%{y}</b><br>Importance: %{x:.0%}<extra></extra>"))
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=280, xaxis_tickformat=".0%", xaxis_title="Relative Importance")
    return fig


@app.callback(
    Output("prediction-result", "children"),
    Input("predict-btn", "n_clicks"),
    State("pred-airline",  "value"),
    State("pred-class",    "value"),
    State("pred-source",   "value"),
    State("pred-dest",     "value"),
    State("pred-dep-time", "value"),
    State("pred-arr-time", "value"),
    State("pred-stops",    "value"),
    State("pred-duration", "value"),
    State("pred-days",     "value"),
    prevent_initial_call=True,
)
def predict_price(n_clicks, airline, seat_class, source, dest, dep_time, arr_time, stops, duration, days_left):
    if not n_clicks:
        return html.Div()

    input_df = pd.DataFrame([{
        "airline": airline, "flight": "UK-706", "source_city": source,
        "departure_time": dep_time, "stops": stops, "arrival_time": arr_time,
        "destination_city": dest, "class": seat_class,
        "duration": float(duration), "days_left": int(days_left), "Unnamed: 0": 0,
    }])

    try:
        predicted = max(0, model.predict(input_df)[0])

        if predicted < 5000:
            badge_color, badge = COLORS["accent3"], "GREAT DEAL"
        elif predicted < 15000:
            badge_color, badge = COLORS["accent"], "AVERAGE PRICE"
        elif predicted < 50000:
            badge_color, badge = "#D97706", "ABOVE AVERAGE"
        else:
            badge_color, badge = COLORS["accent2"], "PREMIUM FARE"

        return card([
            html.Div(style={"textAlign": "center", "padding": "20px 0"}, children=[
                html.Div("PREDICTED PRICE", style={"color": COLORS["subtext"], "fontSize": "11px", "fontWeight": "600", "letterSpacing": "0.10em", "marginBottom": "8px"}),
                html.Div(f"Rs.{predicted:,.0f}", style={"fontFamily": "'Syne', sans-serif", "fontSize": "52px", "fontWeight": "800", "color": COLORS["accent"], "lineHeight": "1.1"}),
                html.Div("INR", style={"color": COLORS["subtext"], "fontSize": "13px", "marginTop": "4px"}),
                html.Div(badge, style={"display": "inline-block", "background": badge_color, "color": "white", "padding": "4px 16px", "borderRadius": "20px", "fontSize": "11px", "fontWeight": "600", "letterSpacing": "0.06em", "marginTop": "14px"}),
            ]),
            html.Hr(style={"borderColor": COLORS["border"], "margin": "16px 0"}),
            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "12px", "textAlign": "center"}, children=[
                html.Div([html.Div(airline,               style={"fontWeight": "600", "fontSize": "13px"}), html.Div("Airline",   style={"color": COLORS["subtext"], "fontSize": "10px", "marginTop": "2px"})]),
                html.Div([html.Div(f"{source} to {dest}", style={"fontWeight": "600", "fontSize": "12px"}), html.Div("Route",     style={"color": COLORS["subtext"], "fontSize": "10px", "marginTop": "2px"})]),
                html.Div([html.Div(seat_class,            style={"fontWeight": "600", "fontSize": "13px"}), html.Div("Class",     style={"color": COLORS["subtext"], "fontSize": "10px", "marginTop": "2px"})]),
                html.Div([html.Div(f"{duration}h",        style={"fontWeight": "600", "fontSize": "13px"}), html.Div("Duration",  style={"color": COLORS["subtext"], "fontSize": "10px", "marginTop": "2px"})]),
                html.Div([html.Div(f"{days_left} days",   style={"fontWeight": "600", "fontSize": "13px"}), html.Div("Days Left", style={"color": COLORS["subtext"], "fontSize": "10px", "marginTop": "2px"})]),
                html.Div([html.Div(stops,                 style={"fontWeight": "600", "fontSize": "13px"}), html.Div("Stops",     style={"color": COLORS["subtext"], "fontSize": "10px", "marginTop": "2px"})]),
            ]),
        ])

    except Exception as e:
        return card([html.Div(f"Prediction error: {str(e)}", style={"color": COLORS["accent2"], "padding": "20px"})])


if __name__ == "__main__":
    app.run(debug=True)
