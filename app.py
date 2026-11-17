from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==========================================================
# STOCK PRICE PREDICTION RESEARCH PLATFORM
# BSc Financial Engineering | University of Colombo
# ==========================================================

st.set_page_config(
    page_title="Stock Price Prediction | University of Colombo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "research_outputs"

STOCKS = {
    "AAPL": {"name": "Apple", "description": "A technology company known for the iPhone, Mac and digital services."},
    "MSFT": {"name": "Microsoft", "description": "A technology company with businesses in software, cloud computing and services."},
    "TSLA": {"name": "Tesla", "description": "A company focused on electric vehicles and energy products."}
}
MODELS = [
    "Naive Baseline", "Random Forest", "XGBoost", "SVR",
    "LSTM", "GRU", "Transformer"
]
MODEL_INFO = {
    "Naive Baseline": "Uses the most recent observed closing price as the estimate for the next observation. This is the simple benchmark used to judge whether more advanced models add value.",
    "Random Forest": "Combines the predictions of many decision trees. Combining trees can help capture relationships in the input data.",
    "XGBoost": "Builds a sequence of decision trees, with later trees attempting to improve on earlier errors.",
    "SVR": "Support Vector Regression is a method that estimates a numerical value while controlling how complex the fitted relationship becomes.",
    "LSTM": "A recurrent neural network designed to learn patterns across sequences of observations.",
    "GRU": "A recurrent neural network that uses gates to manage information across a sequence. It is related to LSTM but uses a different, often more compact structure.",
    "Transformer": "A neural-network architecture that uses attention mechanisms to learn relationships between observations in a sequence."
}
METRICS = {
    "MAE": ("Average absolute error", "The average distance between estimated and observed prices, ignoring whether the estimate was too high or too low. It is measured in US dollars. Lower is better."),
    "RMSE": ("Root mean squared error", "A prediction-error measure that gives larger errors more influence. It is measured in US dollars. Lower is better."),
    "MAPE (%)": ("Average percentage error", "The average absolute error expressed as a percentage of the observed price. Lower is better. It can be unreliable if actual values are zero or close to zero."),
    "R2": ("R-squared", "A comparison with a simple mean-value reference. Values closer to 1 are generally better. A negative value means the model performed worse than that reference on the evaluated data.")
}

# ---- Visual design: responsive text, no fixed-width content ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.block-container { max-width: 1380px; padding-top: 1.4rem; padding-bottom: 3rem; }
h1, h2, h3 { font-family: 'Manrope', sans-serif; color: #152A43; letter-spacing: -0.025em; }
h1 { font-size: clamp(2rem, 4vw, 3.1rem) !important; line-height: 1.12 !important; }
h2 { font-size: clamp(1.35rem, 2.5vw, 1.9rem) !important; }
p, li, label, .stMarkdown { overflow-wrap: anywhere; }
[data-testid="stSidebar"] { background: #F3F6F9; border-right: 1px solid #E1E7EE; }
[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #E1E7EE; padding: 14px 16px; border-radius: 10px; min-width: 0; }
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] { overflow-wrap: anywhere; }
.hero { background: #F1F5F9; border: 1px solid #DCE5EE; border-radius: 14px; padding: clamp(20px, 4vw, 38px); margin: 10px 0 24px 0; }
.kicker { color: #496A88; font-size: .75rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; }
.lead { color: #43566B; font-size: clamp(1rem, 1.6vw, 1.17rem); line-height: 1.7; max-width: 900px; }
.callout { background: #F3F7FB; border-left: 4px solid #356B98; border-radius: 6px; padding: 16px 18px; margin: 12px 0 20px 0; color: #243B53; line-height: 1.65; }
.subtle { color: #63758A; font-size: .9rem; line-height: 1.6; }
.step { border: 1px solid #E1E7EE; border-radius: 10px; padding: 16px 18px; margin: 8px 0; background: #FFFFFF; }
.step-number { color: #356B98; font-weight: 700; font-size: .8rem; letter-spacing: .08em; }
div[data-testid="stDataFrame"] { width: 100%; }
.stButton button, .stDownloadButton button { border-radius: 7px; }
@media (max-width: 700px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  [data-testid="stMetric"] { padding: 10px; }
}
</style>
""", unsafe_allow_html=True)


# ---- Load and validate the supplied research files ----
@st.cache_data
def load_data(folder_string):
    folder = Path(folder_string)
    names = {
        "prices": "historical_stock_prices.csv",
        "returns": "daily_returns.csv",
        "features": "engineered_features.csv",
        "predictions": "final_predictions.csv",
        "comparison": "final_model_comparison.csv",
        "best": "best_models_by_stock.csv",
        "validation": "validation_results.csv"
    }
    missing = [f for f in names.values() if not (folder / f).is_file()]
    if missing:
        raise FileNotFoundError("Missing file(s) in research_outputs: " + ", ".join(missing))

    d = {k: pd.read_csv(folder / v) for k, v in names.items()}
    for key in ("prices", "returns"):
        if "Date" not in d[key].columns:
            raise ValueError(f"{names[key]} must contain a Date column.")
        d[key]["Date"] = pd.to_datetime(d[key]["Date"], errors="coerce")
        d[key] = d[key].dropna(subset=["Date"]).sort_values("Date")
    for key in ("features", "predictions"):
        if "Date" not in d[key].columns:
            raise ValueError(f"{names[key]} must contain a Date column.")
        d[key]["Date"] = pd.to_datetime(d[key]["Date"], errors="coerce")
        d[key] = d[key].dropna(subset=["Date"]).sort_values("Date")
    for key in ("comparison", "best", "validation"):
        if "Stock" not in d[key].columns or "Model" not in d[key].columns:
            raise ValueError(f"{names[key]} must contain Stock and Model columns.")
        for metric in ("MAE", "RMSE", "MAPE (%)", "R2"):
            if metric in d[key].columns:
                d[key][metric] = pd.to_numeric(d[key][metric], errors="coerce")
    return d

try:
    D = load_data(str(DATA))
except Exception as error:
    st.error("The research data could not be loaded.")
    st.write("Check that all seven CSV files are inside a folder named `research_outputs`, next to `app.py`.")
    st.exception(error)
    st.stop()

prices, returns = D["prices"], D["returns"]
features, predictions = D["features"], D["predictions"]
comparison, best, validation = D["comparison"], D["best"], D["validation"]

available_stocks = [s for s in STOCKS if s in prices.columns]
if not available_stocks:
    st.error("The historical price file does not contain AAPL, MSFT or TSLA columns.")
    st.stop()

# ---- Helpers ----
def company_name(ticker):
    return f"{STOCKS[ticker]['name']} ({ticker})"

def price(value):
    return "Not available" if pd.isna(value) else f"${value:,.2f}"

def stock_prices(ticker):
    return prices.set_index("Date")[ticker].dropna().sort_index()

def stock_predictions(ticker):
    return predictions[predictions["Stock"].astype(str).str.upper() == ticker].sort_values("Date").copy()

def stock_results(ticker, source=comparison):
    return source[source["Stock"].astype(str).str.upper() == ticker].copy()

def heading(eyebrow, title, intro):
    st.markdown(f'<div class="kicker">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="lead">{intro}</div>', unsafe_allow_html=True)
    st.write("")

def note(title, text):
    with st.expander(f"{title} — explain this simply"):
        st.write(text)

def choose_stock(key):
    return st.selectbox("Select a company", available_stocks, format_func=company_name, key=key)

def chart_style(fig, height=420):
    fig.update_layout(
        height=height, autosize=True,
        margin=dict(l=12, r=12, t=58, b=12),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="DM Sans", color="#334155"),
        hovermode="x unified", legend_title_text=""
    )
    fig.update_xaxes(showgrid=False, automargin=True)
    fig.update_yaxes(gridcolor="#E9EEF3", automargin=True)
    return fig

def model_rank_table(frame):
    cols = [c for c in ["Model", "MAE", "RMSE", "MAPE (%)", "R2"] if c in frame.columns]
    return frame[cols].sort_values("RMSE").round(4)

# ---- Sidebar ----
with st.sidebar:
    st.markdown("## STOCK PRICE\n## RESEARCH")
    st.caption("An interactive guide to our final-year project")
    st.divider()
    pages = [
        "Welcome",
        "Start with the basics",
        "Our research",
        "Explore the companies",
        "See the predictions",
        "Compare the models",
        "What we discovered",
        "Research data"
    ]
    page = st.radio("Navigate the research", pages, label_visibility="collapsed")
    st.divider()
    st.markdown("**The companies**")
    for ticker in available_stocks:
        st.caption(f"{STOCKS[ticker]['name']} · {ticker}")
    st.divider()
    st.caption("BSc Financial Engineering\nUniversity of Colombo\n2026")

# ==========================================================
# PAGE 1 — WELCOME
# ==========================================================
if page == "Welcome":
    st.markdown('<div class="kicker">UNIVERSITY OF COLOMBO · FINAL-YEAR RESEARCH · 2026</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero"><div class="kicker">A research project made easy to explore</div><h1>Can we predict stock prices more accurately?</h1><div class="lead">We tested seven different approaches to estimate the closing prices of Apple, Microsoft and Tesla. This platform takes you through the question, the data, the methods and what the results actually tell us — one step at a time.</div></div>', unsafe_allow_html=True)

    st.markdown("### The big question")
    st.markdown("**Do advanced machine learning models make better stock-price predictions than a simple method that uses the previous closing price?**")
    st.markdown("You do not need to know anything about the stock market or artificial intelligence to follow this project. Start with the basics, then explore the evidence.")

    ranked = comparison.sort_values("RMSE").groupby("Stock", as_index=False).first()
    baseline_count = int((ranked["Model"] == "Naive Baseline").sum())
    x1, x2, x3 = st.columns(3)
    x1.metric("Companies studied", len(available_stocks))
    x2.metric("Approaches compared", comparison["Model"].nunique())
    x3.metric("Historical price records", f"{len(prices):,}")

    st.markdown("### The headline result")
    st.markdown(
        f'<div class="callout"><strong>The simple benchmark achieved the lowest test RMSE for '
        f'{baseline_count} out of {len(ranked)} companies.</strong><br><br>'
        'That means the advanced approaches did not beat the simple previous-price method '
        'on this study’s final test comparison. It is a useful reminder that a more complicated '
        'model is not automatically a more accurate one.</div>', unsafe_allow_html=True
    )

    st.markdown("### Your route through the project")
    journey = [
        ("01", "Understand the basics", "Learn what a stock, closing price, return and prediction mean."),
        ("02", "Meet the research", "See the research question, objectives, companies and workflow."),
        ("03", "Explore the data", "Look at the actual historical price and return data used in the study."),
        ("04", "Watch the models being tested", "Compare historical estimates with the prices that were observed."),
        ("05", "Understand the evidence", "Compare the models fairly and explore the final findings.")
    ]
    for n, title, description in journey:
        st.markdown(f'<div class="step"><div class="step-number">{n}</div><strong>{title}</strong><br><span class="subtle">{description}</span></div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("**Prepared by**")
    st.write("Bimalya Herath (24SFE037) · Tharuka Hewawasam (24SFE031)")
    st.write("BSc Financial Engineering · Department of Mathematics · Faculty of Science · University of Colombo")
    st.caption("Academic research platform. It is not a live trading service or investment recommendation.")

# ==========================================================
# PAGE 2 — BASICS
# ==========================================================
elif page == "Start with the basics":
    heading("NO PRIOR KNOWLEDGE NEEDED", "First, what are we talking about?", "A few everyday explanations will make the rest of the research much easier to understand.")
    basics = [
        ("What is a share or stock?", "A share represents a small ownership interest in a company. A person who owns a share of Apple, for example, owns a very small part of that company."),
        ("What is a stock price?", "It is the market price at which a share is being traded. The price can move up or down during trading as buyers and sellers respond to information and expectations."),
        ("What is a closing price?", "The recorded price of a share at the end of a trading session. Our research uses historical closing prices as the values the models try to estimate."),
        ("What does prediction mean here?", "A computer model studies patterns in earlier observations and uses them to estimate a price for a later observation. The estimate can be wrong; the research measures how wrong it was."),
        ("What is a daily return?", "It describes the percentage change in price between one trading observation and the next. A positive return means the price increased; a negative return means it decreased."),
        ("What is volatility?", "Volatility describes how much returns vary. If returns move around widely, the stock has had higher historical volatility over that period. It does not tell us exactly what will happen next."),
        ("What is a model?", "A model is a method that uses data to produce an estimate. This project compares seven methods, from a very simple benchmark to more advanced machine learning and deep learning approaches."),
        ("What is prediction error?", "It is the difference between the model's estimate and the price that was actually observed. The closer the estimate is to the observed value, the smaller the error.")
    ]
    for title, body in basics:
        note(title, body)

    st.markdown("### The companies in this study")
    cols = st.columns(len(available_stocks))
    for col, ticker in zip(cols, available_stocks):
        with col:
            st.markdown(f"**{STOCKS[ticker]['name']}**")
            st.markdown(f"`{ticker}`")
            st.write(STOCKS[ticker]["description"])
    st.info("The project studies historical price prediction. It does not decide whether any of these companies are good investments.")

# ==========================================================
# PAGE 3 — RESEARCH
# ==========================================================
elif page == "Our research":
    heading("THE PROJECT, FROM A TO Z", "What did we set out to do?", "This section explains the purpose of the study and the steps followed to compare the prediction approaches.")
    st.markdown("### Why is this research worth doing?")
    st.write("Stock prices change over time. Researchers and market participants have developed many ways to estimate future prices, but a complicated method is not necessarily more accurate than a simple one. This project tests that assumption using three selected companies and a common set of evaluation measures.")
    st.markdown("### Research question")
    st.markdown("> Can machine learning and deep learning models improve stock price prediction compared with a simple baseline approach?")
    st.markdown("### What did we study?")
    st.write("The study examines historical closing prices for Apple (AAPL), Microsoft (MSFT) and Tesla (TSLA). Seven prediction approaches are compared.")
    st.markdown("### The seven approaches")
    categories = {
        "Simple benchmark": ["Naive Baseline"],
        "Machine learning": ["Random Forest", "XGBoost", "SVR"],
        "Deep learning": ["LSTM", "GRU", "Transformer"]
    }
    for category, models in categories.items():
        st.markdown(f"**{category}**")
        for model in models:
            with st.expander(model):
                st.write(MODEL_INFO[model])
    st.markdown("### How the research works")
    steps = [
        ("Historical observations", "Collect the recorded stock-price observations for the selected companies."),
        ("Prepare the data", "Organise the observations so they can be analysed consistently."),
        ("Create additional features", "Calculate variables such as daily returns, moving averages and rolling volatility."),
        ("Develop prediction approaches", "Apply the selected benchmark, machine learning and deep learning methods."),
        ("Generate estimates", "Produce price estimates for the historical test observations."),
        ("Evaluate the estimates", "Compare estimates with observed prices using MAE, RMSE, MAPE and R-squared."),
        ("Interpret the evidence", "Compare results and consider what can and cannot be concluded.")
    ]
    for i, (title, body) in enumerate(steps, 1):
        st.markdown(f'<div class="step"><div class="step-number">STEP {i}</div><strong>{title}</strong><br><span class="subtle">{body}</span></div>', unsafe_allow_html=True)
    start_date = prices["Date"].min()
    end_date = prices["Date"].max()
    st.markdown("### What period does the supplied dataset cover?")
    st.write(f"The historical price file contains observations from **{start_date:%d %B %Y} to {end_date:%d %B %Y}**.")
    st.warning("The CSV outputs do not identify the original data vendor or fully specify the model settings and data-splitting procedure. Those details should be taken from the approved research methodology/report, not guessed.")

# ==========================================================
# PAGE 4 — DATA EXPLORER
# ==========================================================
elif page == "Explore the companies":
    heading("GET TO KNOW THE DATA", "Explore the three companies", "Choose a company and explore the historical prices and daily movements used in the research.")
    ticker = choose_stock("explore_ticker")
    series = stock_prices(ticker)
    latest = series.iloc[-1]
    change = series.iloc[-1] - series.iloc[-2] if len(series) > 1 else np.nan
    change_pct = (change / series.iloc[-2] * 100) if len(series) > 1 and series.iloc[-2] else np.nan
    a, b, c = st.columns(3)
    a.metric("Latest recorded close", price(latest))
    b.metric("Change from previous record", price(change), f"{change_pct:+.2f}%" if pd.notna(change_pct) else None)
    c.metric("Price observations", f"{len(series):,}")
    st.caption(f"Latest date available in this research file: {series.index.max():%d %B %Y}. This is not a live market price.")

    chart_df = series.rename("Closing price").rename_axis("Date").reset_index()
    fig = px.line(chart_df, x="Date", y="Closing price", title=f"{company_name(ticker)} — historical closing price")
    fig.update_traces(line=dict(color="#356B98", width=2.5))
    st.plotly_chart(chart_style(fig, 430), use_container_width=True)
    note("How to read the price chart", "The horizontal axis shows dates. The vertical axis shows the recorded closing price in US dollars. The line connects one trading observation to the next.")

    r = returns[["Date", ticker]].dropna().copy()
    r["Daily return (%)"] = r[ticker] * 100
    left, right = st.columns(2)
    with left:
        fig = px.line(r, x="Date", y="Daily return (%)", title="Daily price change (%)")
        fig.add_hline(y=0, line_dash="dash", line_color="#7B8794")
        st.plotly_chart(chart_style(fig, 340), use_container_width=True)
    with right:
        fig = px.histogram(r, x="Daily return (%)", nbins=45, title="How daily returns are distributed")
        fig.add_vline(x=0, line_dash="dash", line_color="#7B8794")
        st.plotly_chart(chart_style(fig, 340), use_container_width=True)
    note("What does the return chart show?", "It shows the percentage change from one observation to the next. Positive and negative values represent increases and decreases. The histogram groups similar daily changes together.")
    st.markdown("### Additional variables created for the models")
    f = features[features["Stock"].astype(str).str.upper() == ticker].copy()
    display = [c for c in ["Date", "Close", "Daily_Return", "MA_5", "MA_10", "MA_20", "Volatility_10", "Target"] if c in f.columns]
    st.dataframe(f[display].tail(10).round(5), use_container_width=True, hide_index=True)
    note("Moving averages", "MA_5, MA_10 and MA_20 summarise recent closing prices over windows of 5, 10 and 20 observations. They smooth some of the day-to-day movement.")
    note("Rolling volatility", "Volatility_10 is a rolling measure of how much daily returns varied over a 10-observation window. The values in the file are proportions, not percentages.")
    note("Target", "The file includes a Target variable. Its exact construction should be interpreted according to the research methodology.")

# ==========================================================
# PAGE 5 — PREDICTIONS
# ==========================================================
elif page == "See the predictions":
    heading("LOOK AT THE MODEL OUTPUTS", "What did the models predict?", "Compare the prices that were observed with the estimates generated by the different approaches during the historical test period.")
    ticker = choose_stock("prediction_ticker")
    p = stock_predictions(ticker)
    if p.empty:
        st.warning("No prediction rows are available for this company.")
        st.stop()
    model_cols = [m for m in MODELS if m in p.columns]
    default = [m for m in ["Naive Baseline", "GRU"] if m in model_cols] or model_cols[:2]
    selected = st.multiselect("Choose which model lines to display", model_cols, default=default)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p["Date"], y=p["Actual"], name="Observed price", mode="lines", line=dict(width=3, color="#152A43")))
    for model in selected:
        fig.add_trace(go.Scatter(x=p["Date"], y=p[model], name=model, mode="lines", line=dict(width=2)))
    fig.update_layout(title=f"{company_name(ticker)} — observed price compared with model estimates", xaxis_title="Historical test date", yaxis_title="Closing price (USD)")
    st.plotly_chart(chart_style(fig, 470), use_container_width=True)
    st.caption(f"Test observations shown: {p['Date'].min():%d %b %Y} to {p['Date'].max():%d %b %Y}.")
    note("What does this chart mean?", "The observed-price line shows what the closing price actually was on each test date. The other lines show what each model estimated for those same dates. A line closer to the observed price generally indicates closer estimates.")
    note("Are these future forecasts?", "No. These are historical test-period predictions from the supplied research file. They are compared with prices that were observed in the test data. They are not live predictions and do not guarantee future prices.")
    st.markdown("### Look at one date more closely")
    date_values = p["Date"].dt.strftime("%Y-%m-%d").dropna().unique().tolist()
    selected_date = st.select_slider("Choose a test date", options=date_values, value=date_values[-1])
    row = p[p["Date"].dt.strftime("%Y-%m-%d") == selected_date].iloc[0]
    actual = row["Actual"]
    st.metric("Observed closing price", price(actual))
    detail = []
    for model in model_cols:
        estimate = row[model]
        difference = estimate - actual if pd.notna(estimate) else np.nan
        detail.append({
            "Approach": model,
            "Estimated price (USD)": estimate,
            "Difference (USD)": difference,
            "Absolute error (USD)": abs(difference) if pd.notna(difference) else np.nan
        })
    st.dataframe(pd.DataFrame(detail).sort_values("Absolute error (USD)").round(3), use_container_width=True, hide_index=True)

# ==========================================================
# PAGE 6 — MODEL COMPARISON
# ==========================================================
elif page == "Compare the models":
    heading("WHICH APPROACH WAS MORE ACCURATE?", "Compare the seven approaches", "The table and chart show how closely each model's estimates matched the observed prices in the final test comparison.")
    ticker = choose_stock("metric_ticker")
    result = stock_results(ticker, comparison).dropna(subset=["RMSE"]).sort_values("RMSE")
    if result.empty:
        st.warning("No model comparison results were found for this company.")
        st.stop()
    winner = result.iloc[0]
    st.markdown(f'<div class="callout"><strong>Lowest RMSE for {company_name(ticker)}: {winner["Model"]}</strong><br>RMSE: {winner["RMSE"]:.3f} USD. Lower RMSE means smaller errors under this measure.</div>', unsafe_allow_html=True)
    st.markdown("### Full test comparison")
    st.dataframe(model_rank_table(result), use_container_width=True, hide_index=True)
    st.markdown("### Visual comparison")
    metric = st.selectbox("Select a measure", ["RMSE", "MAE", "MAPE (%)", "R2"])
    ordered = result.sort_values(metric, ascending=(metric != "R2"))
    fig = px.bar(ordered, x="Model", y=metric, text_auto=".3f", title=f"{company_name(ticker)} — {metric} by approach")
    fig.update_traces(marker_color="#477DA8")
    st.plotly_chart(chart_style(fig, 400), use_container_width=True)
    if metric == "R2":
        st.caption("For R-squared, values closer to 1 are generally better. Negative values are possible.")
    else:
        st.caption("For this error measure, a smaller value indicates lower measured prediction error.")
    st.markdown("### What do the measures mean?")
    for key, (label, description) in METRICS.items():
        note(f"{key} — {label}", description)
    st.markdown("### Validation results")
    st.write("Validation results are shown separately from the final test results. They have different roles in model development and should not be mixed into one ranking.")
    val = stock_results(ticker, validation)
    if not val.empty:
        st.dataframe(model_rank_table(val), use_container_width=True, hide_index=True)
    else:
        st.info("No validation results were supplied for this company.")

# ==========================================================
# PAGE 7 — FINDINGS
# ==========================================================
elif page == "What we discovered":
    heading("THE ANSWER TO OUR RESEARCH QUESTION", "What did the results tell us?", "The findings below are calculated from the final test comparison supplied with the project.")
    ranked = comparison.dropna(subset=["RMSE"]).sort_values("RMSE").groupby("Stock", as_index=False).first()
    baseline_wins = int((ranked["Model"] == "Naive Baseline").sum())
    st.markdown(f'<div class="callout"><strong>The Naive Baseline achieved the lowest test RMSE for all {baseline_wins} companies in this dataset.</strong><br><br>For Apple, Microsoft and Tesla, using the previous observed closing price as the estimate produced a lower RMSE than the other tested approaches.</div>', unsafe_allow_html=True)
    st.markdown("### Best-performing approach for each company")
    st.dataframe(ranked[["Stock", "Model", "RMSE", "MAE", "MAPE (%)", "R2"]].rename(columns={"Model": "Lowest-RMSE approach", "RMSE": "Test RMSE (USD)"}).round(4), use_container_width=True, hide_index=True)
    st.markdown("### Best advanced approach for each company")
    advanced = comparison[comparison["Model"] != "Naive Baseline"].dropna(subset=["RMSE"])
    advanced_winners = advanced.sort_values("RMSE").groupby("Stock", as_index=False).first()
    st.dataframe(advanced_winners[["Stock", "Model", "RMSE", "MAE", "MAPE (%)", "R2"]].rename(columns={"Model": "Best advanced approach by RMSE"}).round(4), use_container_width=True, hide_index=True)
    st.markdown("### What is the important message?")
    st.write("The most complex approach is not necessarily the most accurate. A simple benchmark is important because it provides a fair reference point: it helps establish whether advanced modelling actually improves the result.")
    st.markdown("### What can we conclude — and what can we not conclude?")
    st.write("These results describe the three selected companies, the data and features supplied, the approaches tested and the final test period. They do not prove that a simple baseline will always outperform advanced models, nor do they establish what will happen to stock prices in the future.")
    st.markdown("### Limitations to keep in mind")
    st.write("- Only three companies are included in this experiment.\n- Results depend on the dataset, feature design, model configurations and evaluation period.\n- Historical performance is not a guarantee of future performance.\n- The output CSV files do not contain all methodological details; the research report should be used for the full data-source, model-setting and data-splitting description.")
    st.info("This is an academic research project, not investment advice.")

# ==========================================================
# PAGE 8 — DATA / APPENDIX
# ==========================================================
elif page == "Research data":
    heading("FOR ACADEMIC REVIEW", "Research data and technical details", "Inspect the supplied output tables and download the main results for further review.")
    overview = []
    for ticker in available_stocks:
        s = stock_prices(ticker)
        pr = stock_predictions(ticker)
        f = features[features["Stock"].astype(str).str.upper() == ticker]
        overview.append({
            "Company": company_name(ticker),
            "Historical observations": len(s),
            "Historical data begins": s.index.min().date(),
            "Historical data ends": s.index.max().date(),
            "Feature rows": len(f),
            "Test prediction rows": len(pr),
            "Test period begins": pr["Date"].min().date() if not pr.empty else None,
            "Test period ends": pr["Date"].max().date() if not pr.empty else None
        })
    st.markdown("### Dataset coverage")
    st.dataframe(pd.DataFrame(overview), use_container_width=True, hide_index=True)
    st.markdown("### Browse an output file")
    tables = {
        "Final model comparison": comparison,
        "Best models by stock": best,
        "Validation results": validation,
        "Final predictions": predictions,
        "Engineered features": features,
        "Daily returns": returns,
        "Historical closing prices": prices
    }
    selected_table = st.selectbox("Select a table", list(tables))
    table = tables[selected_table]
    st.caption(f"{len(table):,} rows · {len(table.columns)} columns")
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.markdown("### Download results")
    downloads = [
        ("Final model comparison", comparison, "final_model_comparison.csv"),
        ("Best models by stock", best, "best_models_by_stock.csv"),
        ("Validation results", validation, "validation_results.csv"),
        ("Final predictions", predictions, "final_predictions.csv"),
        ("Engineered features", features, "engineered_features.csv"),
        ("Daily returns", returns, "daily_returns.csv"),
        ("Historical stock prices", prices, "historical_stock_prices.csv")
    ]
    for label, frame, filename in downloads:
        st.download_button(
            label=f"Download {label}",
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name=filename,
            mime="text/csv",
            key=f"dl_{filename}"
        )

# ---- Consistent footer ----
st.divider()
st.caption("Stock Price Prediction Research Platform · BSc Financial Engineering · Department of Mathematics · Faculty of Science · University of Colombo")
st.caption("All prices and predictions are historical research outputs. They are not live market quotations, investment advice or guaranteed future outcomes.")
