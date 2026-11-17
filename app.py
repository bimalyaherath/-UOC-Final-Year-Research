from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# STOCK PRICE PREDICTION | FINAL-YEAR RESEARCH PLATFORM
# ============================================================

st.set_page_config(
    page_title="Stock Price Prediction | Research Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "research_outputs"

STOCK_NAMES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "TSLA": "Tesla"
}
MODEL_ORDER = [
    "Naive Baseline", "Random Forest", "XGBoost",
    "SVR", "LSTM", "GRU", "Transformer"
]
MODEL_DESCRIPTIONS = {
    "Naive Baseline": "Uses the previous observed closing price as the next price estimate. It is a simple reference point against which more advanced models are compared.",
    "Random Forest": "Combines the outputs of many decision trees. The combined result is intended to capture patterns in the input data.",
    "XGBoost": "Builds decision trees sequentially, with each new tree attempting to improve on errors made by earlier trees.",
    "SVR": "Support Vector Regression estimates a relationship between input features and a numerical target while controlling model complexity.",
    "LSTM": "A type of recurrent neural network designed to learn patterns across sequences of observations.",
    "GRU": "A recurrent neural network architecture that uses gates to manage information across a sequence. It is related to LSTM but has a different, often simpler, structure.",
    "Transformer": "A neural-network architecture that uses attention mechanisms to learn relationships between elements in a sequence."
}
METRIC_HELP = {
    "MAE": ("Mean Absolute Error", "The average size of the prediction error, ignoring whether the prediction was too high or too low. It is expressed in the same unit as the share price (USD). Lower is better."),
    "RMSE": ("Root Mean Squared Error", "A measure of prediction error that gives larger errors more influence. It is expressed in USD. Lower is better."),
    "MAPE (%)": ("Mean Absolute Percentage Error", "The average absolute prediction error expressed as a percentage of the actual price. Lower is better. It can be unsuitable when actual values are zero or very close to zero."),
    "R2": ("R-squared", "Describes how much variation in the observed values is accounted for by the model relative to a mean-value reference. Values closer to 1 are generally better; a negative value means the model performed worse than that reference on the evaluated data.")
}

# Restrained research-oriented visual styling.
st.markdown("""
<style>
    :root { --ink:#172B4D; --muted:#64748B; --line:#E2E8F0; --blue:#245B8F; }
    .block-container { padding-top: 1.7rem; padding-bottom: 3rem; max-width: 1450px; }
    h1, h2, h3 { color: #172B4D; letter-spacing: -0.025em; }
    h1 { font-weight: 700; }
    [data-testid="stSidebar"] { background: #F5F7FA; border-right: 1px solid #E2E8F0; }
    [data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #E2E8F0; padding: 15px 17px; border-radius: 8px; }
    .eyebrow { color:#52708E; text-transform:uppercase; font-size:.74rem; font-weight:700; letter-spacing:.11em; }
    .lead { color:#475569; font-size:1.08rem; line-height:1.65; max-width:950px; }
    .section-note { color:#64748B; font-size:.9rem; line-height:1.55; }
    .finding { background:#F2F6FA; border-left:4px solid #245B8F; padding:15px 18px; border-radius:4px; margin:10px 0 18px 0; }
    .small-label { color:#64748B; font-size:.82rem; }
    div[data-testid="stDataFrame"] { border:1px solid #E2E8F0; border-radius:7px; }
    hr { border-color:#E2E8F0; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_research_data(data_dir: str):
    folder = Path(data_dir)
    required = {
        "historical": "historical_stock_prices.csv",
        "features": "engineered_features.csv",
        "returns": "daily_returns.csv",
        "predictions": "final_predictions.csv",
        "comparison": "final_model_comparison.csv",
        "best": "best_models_by_stock.csv",
        "validation": "validation_results.csv",
    }
    missing = [filename for filename in required.values() if not (folder / filename).exists()]
    if missing:
        raise FileNotFoundError(
            "The following research data files are missing from research_outputs: "
            + ", ".join(missing)
        )

    data = {key: pd.read_csv(folder / filename) for key, filename in required.items()}

    for key in ("historical", "returns"):
        data[key]["Date"] = pd.to_datetime(data[key]["Date"], errors="coerce")
        data[key] = data[key].dropna(subset=["Date"]).sort_values("Date")

    for key in ("features", "predictions"):
        data[key]["Date"] = pd.to_datetime(data[key]["Date"], errors="coerce")
        data[key] = data[key].dropna(subset=["Date"]).sort_values("Date")

    for key in ("comparison", "best", "validation"):
        for col in ("MAE", "RMSE", "MAPE (%)", "R2", "RMSE_Rank"):
            if col in data[key].columns:
                data[key][col] = pd.to_numeric(data[key][col], errors="coerce")

    return data


try:
    D = load_research_data(str(DATA_DIR))
except Exception as exc:
    st.error("The research data could not be loaded.")
    st.exception(exc)
    st.stop()

historical = D["historical"]
features = D["features"]
returns = D["returns"]
predictions = D["predictions"]
comparison = D["comparison"]
best_models = D["best"]
validation = D["validation"]

stocks = [s for s in STOCK_NAMES if s in historical.columns]
if not stocks:
    st.error("No expected stock columns (AAPL, MSFT, TSLA) were found in historical_stock_prices.csv.")
    st.stop()

# ------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------
def money(value):
    return "—" if pd.isna(value) else f"${value:,.2f}"

def pct(value, digits=2):
    return "—" if pd.isna(value) else f"{value:.{digits}f}%"

def stock_label(ticker):
    return f"{STOCK_NAMES.get(ticker, ticker)} ({ticker})"

def page_header(kicker, title, description):
    st.markdown(f'<div class="eyebrow">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="lead">{description}</div>', unsafe_allow_html=True)
    st.write("")

def explain(title, body):
    with st.expander(f"Understand this: {title}"):
        st.write(body)

def get_price_series(ticker):
    return historical.set_index("Date")[ticker].dropna().sort_index()

def get_stock_comparison(ticker):
    return comparison[comparison["Stock"].astype(str).str.upper() == ticker].copy()

def get_stock_predictions(ticker):
    return predictions[predictions["Stock"].astype(str).str.upper() == ticker].copy().sort_values("Date")

def get_best_model(ticker):
    row = best_models[best_models["Stock"].astype(str).str.upper() == ticker]
    return row.iloc[0] if not row.empty else None

def metric_definition(metric):
    if metric in METRIC_HELP:
        name, desc = METRIC_HELP[metric]
        return f"**{name}.** {desc}"
    return ""

def metric_table(df):
    cols = [c for c in ["Model", "MAE", "RMSE", "MAPE (%)", "R2", "RMSE_Rank"] if c in df.columns]
    return df[cols].sort_values("RMSE").round(4)

def add_stock_filter(key="stock_selector"):
    options = stocks
    return st.selectbox(
        "Choose a company",
        options,
        format_func=stock_label,
        key=key
    )

def chart_layout(fig, height=430):
    fig.update_layout(
        height=height,
        margin=dict(l=15, r=15, t=55, b=15),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#334155"),
        hovermode="x unified",
        legend_title_text=""
    )
    fig.update_xaxes(showgrid=False, linecolor="#CBD5E1")
    fig.update_yaxes(gridcolor="#EDF1F5", zerolinecolor="#CBD5E1")
    return fig

# ------------------------------------------------------------
# Navigation
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("## Research Platform")
    st.caption("Stock price prediction | Final-year research")
    st.divider()
    page = st.radio(
        "Explore the research",
        [
            "Start Here",
            "Stock Market Basics",
            "Research Design",
            "Explore the Data",
            "Prediction Explorer",
            "Compare the Models",
            "Research Findings",
            "Technical Appendix",
        ],
        label_visibility="collapsed"
    )
    st.divider()
    st.markdown("**Study scope**")
    st.caption("Apple · Microsoft · Tesla")
    st.caption("2019–2026 historical dataset")
    st.caption("Seven prediction approaches")
    st.divider()
    st.caption("Department of Mathematics\nFaculty of Science\nUniversity of Colombo")

# ------------------------------------------------------------
# 1. START HERE
# ------------------------------------------------------------
if page == "Start Here":
    st.markdown('<div class="eyebrow">UOC · FINAL-YEAR RESEARCH · 2026</div>', unsafe_allow_html=True)
    st.title("Stock Price Prediction")
    st.markdown(
        '<div class="lead">A comparative study of machine learning and deep learning approaches '
        'for predicting the closing prices of Apple, Microsoft and Tesla.</div>',
        unsafe_allow_html=True
    )
    st.write("")
    st.markdown("**The question behind this research**")
    st.markdown(
        "Can more advanced prediction models estimate stock prices more accurately "
        "than a simple approach that uses the previous closing price?"
    )

    # Findings are read from the supplied final test comparison, not hard-coded claims.
    overall_winners = (
        comparison.sort_values("RMSE")
        .groupby("Stock", as_index=False)
        .first()[["Stock", "Model", "RMSE"]]
    )
    baseline_wins = int((overall_winners["Model"] == "Naive Baseline").sum())
    total_stocks = len(overall_winners)
    gru_rows = comparison[comparison["Model"].astype(str).str.upper() == "GRU"]
    advanced = comparison[comparison["Model"] != "Naive Baseline"]
    advanced_best = advanced.sort_values("RMSE").groupby("Stock", as_index=False).first()

    c1, c2, c3 = st.columns(3)
    c1.metric("Companies studied", len(stocks))
    c2.metric("Prediction approaches", comparison["Model"].nunique())
    c3.metric("Historical price observations", f"{len(historical):,}")

    st.markdown("### The main result")
    st.markdown(
        f'<div class="finding"><strong>The simple benchmark achieved the lowest test RMSE '
        f'for {baseline_wins} of {total_stocks} stocks.</strong><br>'
        'In this experiment, the more complex models did not outperform the previous-price '
        'benchmark on the final test comparison. This is an important research result: '
        'model complexity alone does not guarantee more accurate predictions.</div>',
        unsafe_allow_html=True
    )

    cols = st.columns(len(stocks))
    for col, ticker in zip(cols, stocks):
        winner = overall_winners[overall_winners["Stock"] == ticker]
        if not winner.empty:
            col.markdown(f"**{stock_label(ticker)}**")
            col.write(f"Lowest test RMSE: **{winner.iloc[0]['Model']}**")
            col.caption(f"RMSE: {winner.iloc[0]['RMSE']:.3f} USD")

    st.divider()
    st.markdown("### How to use this platform")
    journey = [
        ("01 · Understand", "Start with the Stock Market Basics page if terms such as closing price, return or volatility are unfamiliar."),
        ("02 · Explore", "Review the research question, the selected companies and the historical dataset."),
        ("03 · See predictions", "Choose a company and compare its observed prices with estimates from each model."),
        ("04 · Compare fairly", "Use the evaluation metrics to see how close each model's estimates were to the observed prices."),
        ("05 · Interpret", "Read the findings and limitations before drawing conclusions.")
    ]
    for heading, body in journey:
        with st.container(border=True):
            st.markdown(f"**{heading}**")
            st.write(body)

    st.markdown("### Research team")
    st.write("**Bimalya Herath (24SFE037)** · **Tharuka Hewawasam (24SFE031)**")
    st.write("BSc Financial Engineering · Department of Mathematics · Faculty of Science · University of Colombo")
    st.caption(
        "Research platform for academic and educational use. Historical model results are not "
        "live market data, investment recommendations or guarantees of future prices."
    )

# ------------------------------------------------------------
# 2. STOCK MARKET BASICS
# ------------------------------------------------------------
elif page == "Stock Market Basics":
    page_header(
        "A SHORT INTRODUCTION",
        "Stock market concepts, without the jargon",
        "This page introduces the ideas needed to follow the research. You do not need a finance or programming background."
    )
    concepts = [
        ("What is a stock?", "A stock represents a small ownership share in a company. When someone buys a share of Apple, for example, they own a very small part of that business."),
        ("What is a stock price?", "The stock price is the market price at which a share is being traded. It changes as buyers and sellers place orders and as expectations about the company and wider market change."),
        ("What is a closing price?", "The closing price is the recorded price at the end of a trading session. This research uses historical closing-price observations as the basis for its prediction task."),
        ("What is a prediction model?", "A prediction model is a mathematical or computational method that learns patterns from past observations and uses them to estimate a value. A model's estimate is not a promise about what will happen."),
        ("What is a daily return?", "Daily return describes the percentage change in price from one trading observation to the next. A positive return indicates an increase; a negative return indicates a decrease."),
        ("What is volatility?", "Volatility describes how much returns vary over time. Higher historical volatility means returns have fluctuated more widely in the period being measured. It does not tell us exactly what will happen next."),
        ("What is a model error?", "A model error is the difference between an estimated price and the observed price. Smaller errors generally mean the estimate was closer to what actually happened.")
    ]
    for title, body in concepts:
        explain(title, body)

    st.markdown("### The three companies in this study")
    cols = st.columns(3)
    summaries = {
        "AAPL": "Apple — a technology company known for products such as the iPhone, Mac and related services.",
        "MSFT": "Microsoft — a technology company with businesses in software, cloud computing and related services.",
        "TSLA": "Tesla — a company focused on electric vehicles, energy generation and storage."
    }
    for col, ticker in zip(cols, stocks):
        with col:
            st.markdown(f"**{stock_label(ticker)}**")
            st.write(summaries[ticker])

    st.info("The research compares the historical price-prediction performance of these companies. It does not assess whether their shares should be bought or sold.")

# ------------------------------------------------------------
# 3. RESEARCH DESIGN
# ------------------------------------------------------------
elif page == "Research Design":
    page_header(
        "THE STUDY",
        "What the researchers set out to investigate",
        "The project compares a simple reference method with six machine learning and deep learning approaches using historical stock-price data."
    )
    st.markdown("### Research problem")
    st.write(
        "Stock prices change over time, and their movements can be difficult to estimate. "
        "Different prediction methods may produce different levels of error. The research "
        "therefore evaluates several approaches under the same stock-specific test comparison."
    )
    st.markdown("### Research question")
    st.markdown(
        "> Can machine learning and deep learning models improve stock price prediction "
        "compared with a simple baseline approach?"
    )
    st.markdown("### Companies and study period")
    c1, c2 = st.columns(2)
    c1.metric("Companies", ", ".join(stocks))
    c2.metric("Historical dataset", f"{historical['Date'].min():%d %b %Y} – {historical['Date'].max():%d %b %Y}")

    st.markdown("### The seven approaches")
    model_rows = []
    for model in MODEL_ORDER:
        if model in comparison["Model"].unique():
            model_rows.append({"Approach": model, "Category": "Reference benchmark" if model == "Naive Baseline" else ("Machine learning" if model in ["Random Forest", "XGBoost", "SVR"] else "Deep learning"), "Plain-language description": MODEL_DESCRIPTIONS[model]})
    st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)

    st.markdown("### Research workflow")
    stages = [
        ("1. Historical data", "Use the recorded closing prices for the selected companies."),
        ("2. Data preparation", "Prepare the observations and organise them for analysis."),
        ("3. Feature engineering", "Create additional variables such as daily return, moving averages and rolling volatility."),
        ("4. Model development", "Train and evaluate the selected prediction approaches."),
        ("5. Prediction", "Generate estimates for the held-out historical test observations."),
        ("6. Evaluation", "Compare estimates with observed prices using MAE, RMSE, MAPE and R²."),
        ("7. Interpretation", "Assess what the results show within the scope of this experiment.")
    ]
    for title, body in stages:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(body)

    explain(
        "What is feature engineering?",
        "Feature engineering means creating useful input variables from the original data. "
        "In the supplied research dataset these include daily return, moving averages over "
        "5, 10 and 20 observations, and 10-observation rolling volatility."
    )
    explain(
        "Why include a simple baseline?",
        "A baseline provides a reference level of performance. Without it, a complex model's "
        "error has little context. The baseline in this project is named Naive Baseline."
    )
    st.warning(
        "The supplied output files do not document the original data vendor, exact model "
        "hyperparameters, feature scaling procedure or train/validation/test split rules. "
        "Those details should be added from the research methodology/report rather than inferred here."
    )

# ------------------------------------------------------------
# 4. EXPLORE THE DATA
# ------------------------------------------------------------
elif page == "Explore the Data":
    page_header(
        "THE DATA",
        "Explore the historical observations",
        "Choose a company to inspect its recorded closing prices, daily returns and engineered variables."
    )
    ticker = add_stock_filter("data_stock")
    prices = get_price_series(ticker)
    latest_date = prices.index.max()
    previous_date = prices.index[-2] if len(prices) > 1 else None
    latest_value = prices.iloc[-1]
    change = latest_value - prices.iloc[-2] if len(prices) > 1 else np.nan
    change_pct = (change / prices.iloc[-2] * 100) if len(prices) > 1 and prices.iloc[-2] != 0 else np.nan

    a, b, c = st.columns(3)
    a.metric("Latest available close", money(latest_value))
    b.metric("Change from previous observation", money(change), pct(change_pct))
    c.metric("Observations in dataset", f"{len(prices):,}")
    st.caption(f"Latest date in this research dataset: {latest_date:%d %B %Y}. This is not a live market quote.")

    fig = px.line(
        prices.rename("Closing price").rename_axis("Date").reset_index(),
        x="Date", y="Closing price",
        labels={"Closing price": "Closing price (USD)"},
        title=f"{stock_label(ticker)} — historical closing prices"
    )
    st.plotly_chart(chart_layout(fig, 440), use_container_width=True)
    explain(
        "How should I read this chart?",
        "Each point represents a recorded closing price on a trading date. The line connects "
        "observations in chronological order. The vertical axis is measured in US dollars. "
        "A rising section indicates that the recorded price increased over those observations; "
        "a falling section indicates a decrease."
    )

    ret = returns[["Date", ticker]].dropna().copy()
    ret["Return (%)"] = ret[ticker] * 100
    left, right = st.columns(2)
    with left:
        fig_ret = px.line(ret, x="Date", y="Return (%)", title="Daily percentage returns")
        fig_ret.add_hline(y=0, line_dash="dash", line_color="#64748B")
        st.plotly_chart(chart_layout(fig_ret, 350), use_container_width=True)
    with right:
        fig_hist = px.histogram(ret, x="Return (%)", nbins=45, title="Distribution of daily returns")
        fig_hist.add_vline(x=0, line_dash="dash", line_color="#64748B")
        st.plotly_chart(chart_layout(fig_hist, 350), use_container_width=True)
    explain(
        "What does the return distribution show?",
        "The histogram groups daily percentage returns into ranges. Taller bars indicate "
        "ranges that occurred more often in the dataset. The chart helps describe the observed "
        "distribution; it does not predict the next return."
    )

    st.markdown("### Engineered variables")
    stock_features = features[features["Stock"].astype(str).str.upper() == ticker].copy()
    display_cols = [c for c in ["Date", "Close", "Daily_Return", "MA_5", "MA_10", "MA_20", "Volatility_10", "Target"] if c in stock_features.columns]
    st.dataframe(stock_features[display_cols].tail(12).round(5), use_container_width=True, hide_index=True)
    explain("Moving averages (MA_5, MA_10 and MA_20)", "A moving average is the average closing price over a recent window of observations. It smooths short-term fluctuations and provides a summary of recent price levels.")
    explain("Volatility_10", "This field represents a rolling measure of variation in daily returns over a 10-observation window. Values are stored as proportions in the supplied file; the table shows the underlying values.")
    explain("Target", "The supplied engineered-features file includes a Target column. Its precise construction should be described using the project's modelling methodology; the dashboard does not assume a definition beyond the stored field.")

# ------------------------------------------------------------
# 5. PREDICTION EXPLORER
# ------------------------------------------------------------
elif page == "Prediction Explorer":
    page_header(
        "MODEL OUTPUTS",
        "See how close each model came",
        "Compare model estimates with the observed closing prices during the historical test period."
    )
    ticker = add_stock_filter("prediction_stock")
    pred = get_stock_predictions(ticker)
    if pred.empty:
        st.warning("No prediction observations are available for this company.")
        st.stop()

    available_models = [m for m in MODEL_ORDER if m in pred.columns]
    default_models = ["Naive Baseline", "GRU"] if all(m in available_models for m in ["Naive Baseline", "GRU"]) else available_models[:2]
    selected = st.multiselect("Prediction lines to display", available_models, default=default_models)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred["Date"], y=pred["Actual"], mode="lines", name="Observed price", line=dict(width=3, color="#172B4D")))
    for model in selected:
        fig.add_trace(go.Scatter(x=pred["Date"], y=pred[model], mode="lines", name=model, line=dict(width=1.8)))
    fig.update_layout(title=f"{stock_label(ticker)} — observed and estimated prices", xaxis_title="Test-period date", yaxis_title="Closing price (USD)")
    st.plotly_chart(chart_layout(fig, 480), use_container_width=True)

    st.caption(f"Historical test observations: {pred['Date'].min():%d %b %Y} to {pred['Date'].max():%d %b %Y} · {len(pred):,} rows for this company.")
    explain("What am I looking at?", "The dark line is the observed closing price. Each other line is the estimate generated by one selected model for the same test dates. Lines that remain closer to the observed line generally indicate closer estimates, but use the numerical evaluation metrics for a fairer comparison.")
    explain("Does this show future prices?", "No. These are historical test-period predictions stored in the research outputs. They compare model estimates with prices that were subsequently observed in the evaluation dataset. They are not live forecasts or guarantees about future market prices.")

    st.markdown("### Inspect an individual test date")
    dates = pred["Date"].dropna().sort_values().dt.strftime("%Y-%m-%d").unique().tolist()
    chosen_date = st.selectbox("Choose a test date", dates, index=len(dates)-1)
    row = pred[pred["Date"].dt.strftime("%Y-%m-%d") == chosen_date].iloc[0]
    observed = row["Actual"]
    rows = []
    for model in available_models:
        estimate = row[model]
        error = estimate - observed if pd.notna(estimate) else np.nan
        rows.append({
            "Model": model,
            "Estimated price (USD)": estimate,
            "Difference from observed (USD)": error,
            "Absolute difference (USD)": abs(error) if pd.notna(error) else np.nan
        })
    st.metric("Observed closing price on selected date", money(observed))
    st.dataframe(pd.DataFrame(rows).sort_values("Absolute difference (USD)").round(3), use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# 6. COMPARE MODELS
# ------------------------------------------------------------
elif page == "Compare the Models":
    page_header(
        "EVALUATION",
        "Compare the seven prediction approaches",
        "The metrics summarise how far each model's estimates were from the observed prices in the final test comparison."
    )
    ticker = add_stock_filter("comparison_stock")
    results = get_stock_comparison(ticker).sort_values("RMSE")
    if results.empty:
        st.warning("No model comparison results are available for this company.")
        st.stop()

    winner = results.iloc[0]
    st.markdown(
        f'<div class="finding"><strong>Lowest test RMSE for {stock_label(ticker)}: '
        f'{winner["Model"]}.</strong><br>RMSE = {winner["RMSE"]:.3f} USD. '
        'A lower RMSE indicates smaller prediction errors under this metric.</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Evaluation results")
    st.dataframe(metric_table(results), use_container_width=True, hide_index=True)
    explain("Why are there four metrics?", "No single metric describes every aspect of prediction error. MAE gives the average absolute error, RMSE gives greater weight to larger errors, MAPE expresses error as a percentage, and R² compares the model with a mean-value reference.")
    for metric in ["MAE", "RMSE", "MAPE (%)", "R2"]:
        explain(metric, metric_definition(metric))

    metric_choice = st.selectbox("Choose a metric to visualise", ["RMSE", "MAE", "MAPE (%)", "R2"])
    fig = px.bar(
        results.sort_values(metric_choice, ascending=(metric_choice != "R2")),
        x="Model", y=metric_choice,
        title=f"{stock_label(ticker)} — {metric_choice} by model",
        text_auto=".3f"
    )
    fig.update_traces(marker_color="#477DA8")
    st.plotly_chart(chart_layout(fig, 410), use_container_width=True)
    if metric_choice == "R2":
        st.caption("For R², values closer to 1 are generally preferable. Negative values are possible and indicate performance below the mean-value reference.")
    else:
        st.caption("For this error metric, smaller values indicate lower measured prediction error.")

    st.markdown("### Validation versus final test")
    st.write(
        "Validation results and final test results serve different roles in model development. "
        "The validation table is shown separately from the final test comparison; the two should "
        "not be combined into one ranking."
    )
    val = validation[validation["Stock"].astype(str).str.upper() == ticker].sort_values("RMSE")
    if not val.empty:
        st.dataframe(metric_table(val), use_container_width=True, hide_index=True)
    else:
        st.info("No validation rows were found for this company.")

# ------------------------------------------------------------
# 7. RESEARCH FINDINGS
# ------------------------------------------------------------
elif page == "Research Findings":
    page_header(
        "INTERPRETATION",
        "What did the experiment find?",
        "The statements below are calculated from the supplied final test comparison. They describe this experiment, not every stock or every market condition."
    )
    ranked = comparison.sort_values("RMSE").groupby("Stock", as_index=False).first()
    wins = int((ranked["Model"] == "Naive Baseline").sum())
    st.markdown(
        f'<div class="finding"><strong>The Naive Baseline ranked first by RMSE for '
        f'{wins} out of {len(ranked)} studied companies.</strong><br>'
        'For this dataset and test period, using the previous observed price as the estimate '
        'produced lower RMSE than the other approaches.</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Lowest-RMSE approach for each company")
    st.dataframe(ranked.rename(columns={"Model": "Lowest-RMSE approach", "RMSE": "Test RMSE (USD)"}), use_container_width=True, hide_index=True)

    st.markdown("### How did the advanced models compare?")
    advanced = comparison[comparison["Model"] != "Naive Baseline"].copy()
    advanced_winners = advanced.sort_values("RMSE").groupby("Stock", as_index=False).first()
    st.dataframe(advanced_winners[["Stock", "Model", "RMSE", "MAE", "MAPE (%)", "R2"]].rename(columns={"Model": "Best advanced approach by RMSE"}).round(4), use_container_width=True, hide_index=True)

    st.markdown("### Findings supported by the test results")
    st.write(
        "1. The simple Naive Baseline achieved the lowest RMSE for each of the three stocks in the supplied final comparison."
    )
    st.write(
        "2. Among the advanced approaches, the best-performing model was not identical for every stock. "
        "The table above identifies the lowest-RMSE advanced approach for each company."
    )
    st.write(
        "3. The results show that greater model complexity did not automatically translate into lower prediction error in this experiment."
    )

    st.markdown("### What should we take away?")
    st.write(
        "A sophisticated model should not be assumed to be better simply because it is more complex. "
        "A simple benchmark is essential: it shows whether added modelling effort actually improves "
        "performance. In this study, the baseline was difficult to outperform on the final test data."
    )

    st.markdown("### Scope and limitations")
    st.write(
        "- The experiment covers three selected companies: Apple, Microsoft and Tesla.\n"
        "- The conclusions apply to the supplied dataset, engineered variables, modelling choices and evaluation period.\n"
        "- Historical prediction performance does not guarantee future performance.\n"
        "- The output files alone do not provide all methodological details, including the data vendor, exact model settings and full split protocol. These should be documented in the accompanying research report."
    )
    st.info("This platform presents academic findings. It is not an investment recommendation or a trading system.")

# ------------------------------------------------------------
# 8. TECHNICAL APPENDIX
# ------------------------------------------------------------
elif page == "Technical Appendix":
    page_header(
        "FOR RESEARCHERS AND EVALUATORS",
        "Technical appendix",
        "This section provides direct access to the supplied research outputs and key dataset details."
    )
    st.markdown("### Dataset coverage")
    summary = []
    for ticker in stocks:
        p = get_price_series(ticker)
        f = features[features["Stock"].astype(str).str.upper() == ticker]
        pr = get_stock_predictions(ticker)
        summary.append({
            "Stock": ticker,
            "Historical observations": len(p),
            "Historical start": p.index.min().date(),
            "Historical end": p.index.max().date(),
            "Feature rows": len(f),
            "Test prediction rows": len(pr),
            "Test start": pr["Date"].min().date() if not pr.empty else None,
            "Test end": pr["Date"].max().date() if not pr.empty else None
        })
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

    st.markdown("### Supplied output tables")
    datasets = {
        "Final test model comparison": comparison,
        "Best model by stock": best_models,
        "Validation results": validation,
        "Engineered features": features,
        "Final predictions": predictions,
    }
    selected_table = st.selectbox("Choose a research output", list(datasets.keys()))
    table = datasets[selected_table]
    st.caption(f"{len(table):,} rows · {len(table.columns)} columns")
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.markdown("### Download research outputs")
    for label, frame, filename in [
        ("Final model comparison", comparison, "final_model_comparison.csv"),
        ("Best models by stock", best_models, "best_models_by_stock.csv"),
        ("Validation results", validation, "validation_results.csv"),
        ("Final predictions", predictions, "final_predictions.csv"),
        ("Engineered features", features, "engineered_features.csv"),
    ]:
        st.download_button(
            f"Download {label}",
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name=filename,
            mime="text/csv",
            key=f"download_{filename}"
        )

    st.markdown("### Metric definitions")
    for metric, (name, description) in METRIC_HELP.items():
        st.markdown(f"**{metric} — {name}**")
        st.write(description)

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.divider()
st.caption(
    "Stock Price Prediction Research Platform · BSc Financial Engineering · "
    "Department of Mathematics · Faculty of Science · University of Colombo"
)
st.caption(
    "Academic research use only. Prices and predictions are historical dataset outputs, "
    "not real-time quotes, investment advice or guaranteed future outcomes."
)
