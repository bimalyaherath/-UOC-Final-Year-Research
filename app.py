
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Stock Prediction Research Dashboard",
    page_icon="📈",
    layout="wide"
)

# --------------------------------------------------
# LOAD RESEARCH DATA
# --------------------------------------------------

DATA_DIR = "research_outputs"

@st.cache_data
def load_data():

    historical = pd.read_csv(
        f"{DATA_DIR}/historical_stock_prices.csv",
        index_col=0,
        parse_dates=True
    )

    features = pd.read_csv(
        f"{DATA_DIR}/engineered_features.csv",
        parse_dates=["Date"]
    )

    returns = pd.read_csv(
        f"{DATA_DIR}/daily_returns.csv",
        index_col=0,
        parse_dates=True
    )

    predictions = pd.read_csv(
        f"{DATA_DIR}/final_predictions.csv",
        parse_dates=["Date"]
    )

    comparison = pd.read_csv(
        f"{DATA_DIR}/final_model_comparison.csv"
    )

    best_models = pd.read_csv(
        f"{DATA_DIR}/best_models_by_stock.csv"
    )

    validation = pd.read_csv(
        f"{DATA_DIR}/validation_results.csv"
    )

    return (
        historical,
        features,
        returns,
        predictions,
        comparison,
        best_models,
        validation
    )


(
    historical,
    features,
    returns,
    predictions,
    comparison,
    best_models,
    validation
) = load_data()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("📊 Research Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Market Overview",
        "Stock Analysis",
        "Prediction Analysis",
        "Model Comparison",
        "Investment Insights",
        "Research Findings"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Academic research project | Stock Price Forecasting"
)

# --------------------------------------------------
# DASHBOARD TITLE
# --------------------------------------------------

st.title("📈 Stock Price Prediction Research Dashboard")

st.markdown("""
<div style="text-align: center; line-height: 1.7;">

<h4>BSc Financial Engineering</h4>

<p>
Department of Mathematics<br>
Faculty of Science<br>
University of Colombo
</p>

<h4>Final Year Project</h4>

<p>
<strong>Bimalya Herath</strong> (24sfe037)<br>
<strong>Tharuka Hewawasam</strong> (24sfe031)
</p>

<hr>

</div>
""", unsafe_allow_html=True)

st.caption(
    "Comparative analysis of machine learning and deep learning models "
    "for AAPL, MSFT and TSLA."
)

# --------------------------------------------------
# MARKET OVERVIEW
# --------------------------------------------------

if page == "Market Overview":

    st.header("Market Overview")

    selected_stock = st.selectbox(
        "Select Stock",
        ["AAPL", "MSFT", "TSLA"]
    )

    price_series = historical[selected_stock].dropna()

    latest_price = price_series.iloc[-1]

    previous_price = price_series.iloc[-2]

    price_change = latest_price - previous_price

    percentage_change = (
        price_change / previous_price
    ) * 100

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Latest Closing Price",
        f"${latest_price:.2f}"
    )

    col2.metric(
        "Daily Price Change",
        f"${price_change:.2f}",
        f"{percentage_change:.2f}%"
    )

    col3.metric(
        "Historical Observations",
        f"{len(price_series):,}"
    )

    st.subheader("Historical Closing Price")

    price_chart = px.line(
        price_series,
        title=f"{selected_stock} Historical Closing Prices",
        labels={
            "index": "Date",
            "value": "Closing Price (USD)"
        }
    )

    st.plotly_chart(price_chart, use_container_width=True)

    st.subheader("Stock Price Summary")

    st.dataframe(
        price_series.describe().to_frame(
            name="Closing Price (USD)"
        ),
        use_container_width=True
    )

# --------------------------------------------------
# STOCK ANALYSIS
# --------------------------------------------------

elif page == "Stock Analysis":

    st.header("Individual Stock Analysis")

    selected_stock = st.selectbox(
        "Select Stock",
        ["AAPL", "MSFT", "TSLA"]
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Daily Returns")

        return_chart = px.line(
            returns,
            y=selected_stock,
            title=f"{selected_stock} Daily Returns"
        )

        st.plotly_chart(
            return_chart,
            use_container_width=True
        )

    with col2:

        st.subheader("Return Distribution")

        return_chart = px.histogram(
            returns,
            x=selected_stock,
            nbins=50,
            title=f"{selected_stock} Return Distribution"
        )

        st.plotly_chart(
            return_chart,
            use_container_width=True
        )

    stock_features = features[
        features["Stock"] == selected_stock
    ]

    st.subheader("Engineered Features")

    feature_columns = [
        "Close",
        "Daily_Return",
        "MA_5",
        "MA_10",
        "MA_20",
        "Volatility_10"
    ]

    st.dataframe(
        stock_features[["Date"] + feature_columns].tail(10),
        use_container_width=True
    )

# --------------------------------------------------
# PREDICTION ANALYSIS
# --------------------------------------------------

elif page == "Prediction Analysis":

    st.header("Actual vs Predicted Prices")

    selected_stock = st.selectbox(
        "Select Stock",
        ["AAPL", "MSFT", "TSLA"]
    )

    stock_predictions = predictions[
        predictions["Stock"] == selected_stock
    ].copy()

    stock_predictions = stock_predictions.sort_values("Date")

    model_options = [
        "Naive Baseline",
        "Random Forest",
        "XGBoost",
        "SVR",
        "LSTM",
        "GRU",
        "Transformer"
    ]

    selected_models = st.multiselect(
        "Select Models",
        model_options,
        default=model_options
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=stock_predictions["Date"],
            y=stock_predictions["Actual"],
            mode="lines",
            name="Actual Price",
            line=dict(width=3)
        )
    )

    for model in selected_models:

        if model in stock_predictions.columns:

            fig.add_trace(
                go.Scatter(
                    x=stock_predictions["Date"],
                    y=stock_predictions[model],
                    mode="lines",
                    name=model
                )
            )

    fig.update_layout(
        title=f"{selected_stock}: Actual vs Predicted Prices",
        xaxis_title="Date",
        yaxis_title="Closing Price (USD)",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Predictions represent model estimates and are "
        "not guaranteed future prices."
    )

# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------

elif page == "Model Comparison":

    st.header("Model Performance Comparison")

    selected_stock = st.selectbox(
        "Select Stock",
        ["AAPL", "MSFT", "TSLA"]
    )

    stock_results = comparison[
        comparison["Stock"] == selected_stock
    ].sort_values("RMSE")

    st.subheader("Performance Metrics")

    st.dataframe(
        stock_results[
            ["Model", "MAE", "RMSE", "MAPE (%)", "R2"]
        ].round(4),
        use_container_width=True
    )

    st.subheader("RMSE Comparison")

    rmse_chart = px.bar(
        stock_results,
        x="Model",
        y="RMSE",
        color="Model",
        title=f"{selected_stock} Model RMSE Comparison"
    )

    st.plotly_chart(
        rmse_chart,
        use_container_width=True
    )

    best = best_models[
        best_models["Stock"] == selected_stock
    ]

    if not best.empty:

        st.success(
            f"Best-performing approach on the test set: "
            f"{best.iloc[0]['Model']}"
        )

# --------------------------------------------------
# INVESTMENT INSIGHTS
# --------------------------------------------------

elif page == "Investment Insights":

    st.header("Risk and Performance Analysis")

    selected_stock = st.selectbox(
        "Select Stock",
        ["AAPL", "MSFT", "TSLA"]
    )

    stock_returns = returns[selected_stock].dropna()

    annualised_volatility = (
        stock_returns.std() * np.sqrt(252) * 100
    )

    average_daily_return = stock_returns.mean() * 100

    col1, col2 = st.columns(2)

    col1.metric(
        "Annualised Historical Volatility",
        f"{annualised_volatility:.2f}%"
    )

    col2.metric(
        "Average Daily Return",
        f"{average_daily_return:.3f}%"
    )

    st.subheader("Historical Risk Profile")

    st.write(
        "Historical volatility measures the variability "
        "of daily returns. It does not guarantee future risk."
    )

    volatility_chart = px.line(
        features[features["Stock"] == selected_stock],
        x="Date",
        y="Volatility_10",
        title=f"{selected_stock} Rolling 10-Day Volatility"
    )

    st.plotly_chart(
        volatility_chart,
        use_container_width=True
    )

    st.warning(
        "This section is for academic and educational "
        "analysis only. It does not provide investment advice."
    )

# --------------------------------------------------
# RESEARCH FINDINGS
# --------------------------------------------------

elif page == "Research Findings":

    st.header("Research Findings")

    st.subheader("Final Test Performance")

    st.dataframe(
        comparison.sort_values(
            ["Stock", "RMSE"]
        ).round(4),
        use_container_width=True
    )

    st.subheader("Best Test-Performing Approach")

    st.dataframe(
        best_models,
        use_container_width=True
    )

    st.subheader("Validation Results")

    st.dataframe(
        validation.round(4),
        use_container_width=True
    )

    st.subheader("Key Research Observation")

    st.markdown(
        """
        - The Naïve Persistence Baseline achieved the lowest
          test RMSE across all three stocks.
        - GRU was the strongest of the six machine-learning
          and deep-learning models in the test comparison.
        - Model performance differed across stocks.
        - Greater model complexity did not necessarily
          produce better forecasting accuracy.
        """
    )

    st.info(
        "These findings apply to the selected stocks, "
        "features, evaluation period and experimental setup."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "UOC Final Year Research | Stock Price Prediction "
    "Using Machine Learning and Deep Learning"
)
