import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Stock Insights | Research Dashboard",
    page_icon="📈",
    layout="wide"
)

# ==================================================
# 2. CUSTOM STYLING
# ==================================================

st.markdown("""
<style>

    .main {
        background-color: #F8FAFC;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1 {
        color: #17365D;
        font-weight: 700;
    }

    h2, h3 {
        color: #24476B;
    }

    .info-box {
        background-color: #EAF2F8;
        padding: 18px;
        border-radius: 10px;
        border-left: 5px solid #2874A6;
        margin-bottom: 20px;
    }

    .success-box {
        background-color: #E9F7EF;
        padding: 18px;
        border-radius: 10px;
        border-left: 5px solid #239B56;
    }

    .small-text {
        color: #566573;
        font-size: 14px;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        padding: 18px;
        border-radius: 10px;
        border: 1px solid #E5E7EB;
    }

</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. LOAD RESEARCH DATA
# ==================================================

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


try:

    (
        historical,
        features,
        returns,
        predictions,
        comparison,
        best_models,
        validation
    ) = load_data()

except FileNotFoundError:

    st.error(
        "Research data files could not be found. "
        "Please check the research_outputs folder."
    )

    st.stop()


# ==================================================
# 4. GENERAL SETTINGS
# ==================================================

STOCKS = ["AAPL", "MSFT", "TSLA"]

STOCK_NAMES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "TSLA": "Tesla"
}


# ==================================================
# 5. SIDEBAR NAVIGATION
# ==================================================

st.sidebar.title("📈 Stock Insights")

st.sidebar.caption(
    "Making stock market predictions easier to understand."
)

st.sidebar.markdown("---")

page = st.sidebar.radio(

    "Explore Dashboard",

    [
        "🏠 Home",
        "📊 Explore Stocks",
        "🔮 Price Predictions",
        "🏆 Model Performance",
        "💡 Risk & Returns",
        "🎓 Research Results"
    ]

)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **About this project**

    This dashboard presents the results of a research
    project comparing different machine learning and
    deep learning models for stock price prediction.
    """
)

st.sidebar.caption(
    "University of Colombo | Final Year Research"
)


# ==================================================
# 6. MAIN HEADER
# ==================================================

st.title("📈 Stock Insights")

st.markdown(
    """
    ### Understand stock prices. Explore predictions.
    Discover how different AI models perform.

    This dashboard analyses Apple, Microsoft and Tesla
    using historical stock market data.
    """
)

st.markdown("---")


# ==================================================
# 7. HOME PAGE
# ==================================================

if page == "🏠 Home":

    st.header("Welcome to Stock Insights!")

    st.markdown("""
    <div class="info-box">

    <h3>What can you discover here?</h3>

    This dashboard helps you understand how stock prices
    change over time and how different computer models
    attempt to predict future prices.

    No financial or technical background is required.

    </div>
    """, unsafe_allow_html=True)


    # STOCK OVERVIEW

    st.subheader("Explore the Three Companies")

    cols = st.columns(3)

    for col, stock in zip(cols, STOCKS):

        price = historical[stock].dropna()

        latest_price = price.iloc[-1]

        previous_price = price.iloc[-2]

        change = latest_price - previous_price

        percentage = (change / previous_price) * 100

        with col:

            st.metric(
                label=STOCK_NAMES[stock],
                value=f"${latest_price:.2f}",
                delta=f"{percentage:.2f}%"
            )

            st.caption(
                f"Ticker symbol: {stock}"
            )


    st.markdown("---")


    # EXPLAIN THE PROJECT

    st.subheader("How Does This Research Work?")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("### 1️⃣")

        st.markdown("**Collect Market Data**")

        st.write(
            "We collect historical stock prices "
            "from Yahoo Finance."
        )

    with col2:

        st.markdown("### 2️⃣")

        st.markdown("**Make Predictions**")

        st.write(
            "We use seven different approaches "
            "to estimate stock prices."
        )

    with col3:

        st.markdown("### 3️⃣")

        st.markdown("**Compare Performance**")

        st.write(
            "We compare the predictions with "
            "actual stock prices to identify "
            "which approaches performed better."
        )


    st.markdown("---")


    # DATA SUMMARY

    st.subheader("About Our Research Data")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Companies Analysed",
        "3"
    )

    col2.metric(
        "Prediction Approaches",
        "7"
    )

    col3.metric(
        "Historical Records",
        f"{len(historical):,}"
    )


    st.info(
        "The dashboard presents historical research results. "
        "Predictions are estimates, not guaranteed future prices."
    )


# ==================================================
# 8. EXPLORE STOCKS
# ==================================================

elif page == "📊 Explore Stocks":

    st.header("Explore Individual Stocks")

    st.write(
        "Choose a company to explore how its stock price "
        "has changed over time."
    )

    selected_stock = st.selectbox(
        "Choose a Company",
        STOCKS,
        format_func=lambda x: STOCK_NAMES[x]
    )

    price_series = historical[selected_stock].dropna()

    returns_series = returns[selected_stock].dropna()


    # PRICE INFORMATION

    latest_price = price_series.iloc[-1]

    previous_price = price_series.iloc[-2]

    change = latest_price - previous_price

    percentage = (change / previous_price) * 100

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Latest Closing Price",
        f"${latest_price:.2f}"
    )

    col2.metric(
        "Change from Previous Day",
        f"${change:.2f}",
        f"{percentage:.2f}%"
    )

    col3.metric(
        "Available Trading Days",
        f"{len(price_series):,}"
    )


    st.markdown("---")


    # HISTORICAL PRICE CHART

    st.subheader("How Has the Stock Price Changed?")

    st.caption(
        "This chart shows the company's historical closing "
        "price. Move your cursor over the chart to see "
        "individual dates and prices."
    )

    fig = px.line(

        x=price_series.index,
        y=price_series.values,

        labels={
            "x": "Date",
            "y": "Stock Price (USD)"
        },

        title=f"{STOCK_NAMES[selected_stock]} Stock Price History"

    )

    fig.update_layout(
        hovermode="x unified",
        template="plotly_white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # DAILY RETURNS

    st.markdown("---")

    st.subheader("How Much Does the Price Change Each Day?")

    st.write(
        "The following chart shows the percentage change "
        "in the stock price from one trading day to the next."
    )

    return_fig = px.line(

        x=returns_series.index,
        y=returns_series.values * 100,

        labels={
            "x": "Date",
            "y": "Daily Price Change (%)"
        },

        title="Daily Percentage Price Changes"

    )

    return_fig.add_hline(
        y=0,
        line_dash="dash"
    )

    return_fig.update_layout(
        template="plotly_white"
    )

    st.plotly_chart(
        return_fig,
        use_container_width=True
    )


    with st.expander("💡 How should I understand this chart?"):

        st.write(
            """
            - Above zero: The stock price increased that day.
            - Below zero: The stock price decreased that day.
            - Close to zero: The stock price changed very little.

            Larger movements indicate greater day-to-day
            price fluctuations.
            """
        )


    # PRICE DISTRIBUTION

    st.markdown("---")

    st.subheader("How Frequently Did Different Price Changes Occur?")

    distribution_fig = px.histogram(

        x=returns_series * 100,

        nbins=50,

        labels={
            "x": "Daily Price Change (%)",
            "y": "Number of Trading Days"
        },

        title="Distribution of Daily Price Changes"

    )

    distribution_fig.update_layout(
        template="plotly_white"
    )

    st.plotly_chart(
        distribution_fig,
        use_container_width=True
    )


# ==================================================
# 9. PRICE PREDICTIONS
# ==================================================

elif page == "🔮 Price Predictions":

    st.header("How Accurate Are the Predictions?")

    st.markdown("""
    Here, you can compare the actual stock prices with
    the prices estimated by different computer models.

    The closer a prediction is to the actual price,
    the more accurate that prediction is.
    """)


    selected_stock = st.selectbox(
        "Choose a Company",
        STOCKS,
        format_func=lambda x: STOCK_NAMES[x]
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

    available_models = [
        model for model in model_options
        if model in stock_predictions.columns
    ]


    selected_models = st.multiselect(

        "Choose prediction approaches to compare",

        available_models,

        default=available_models

    )


    # PREDICTION GRAPH

    fig = go.Figure()


    fig.add_trace(

        go.Scatter(

            x=stock_predictions["Date"],

            y=stock_predictions["Actual"],

            mode="lines",

            name="Actual Stock Price",

            line=dict(
                width=4
            )

        )

    )


    for model in selected_models:

        fig.add_trace(

            go.Scatter(

                x=stock_predictions["Date"],

                y=stock_predictions[model],

                mode="lines",

                name=model

            )

        )


    fig.update_layout(

        title=f"{STOCK_NAMES[selected_stock]}: Actual vs Predicted Prices",

        xaxis_title="Date",

        yaxis_title="Stock Price (USD)",

        hovermode="x unified",

        template="plotly_white"

    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.markdown("---")


    # MODEL EXPLANATION

    st.subheader("What Do These Model Names Mean?")

    model_explanations = {

        "Naive Baseline":
        "Uses the last observed price as the prediction.",

        "Random Forest":
        "Combines multiple decision trees to produce a prediction.",

        "XGBoost":
        "Builds a sequence of decision trees to improve prediction accuracy.",

        "SVR":
        "Uses Support Vector Regression to estimate stock prices.",

        "LSTM":
        "A deep learning model designed to learn patterns from sequential data.",

        "GRU":
        "A recurrent deep learning model that learns patterns in sequential data.",

        "Transformer":
        "A deep learning architecture that uses attention mechanisms to learn relationships in data."

    }


    with st.expander("Understand the seven prediction approaches"):

        for model, explanation in model_explanations.items():

            st.markdown(f"**{model}**")

            st.write(explanation)

            st.markdown("---")


    st.info(
        "Important: These are historical test-period predictions. "
        "They do not guarantee what the stock price will be in the future."
    )


# ==================================================
# 10. MODEL PERFORMANCE
# ==================================================

elif page == "🏆 Model Performance":

    st.header("Which Prediction Approach Performed Best?")

    st.write(
        "We evaluate each approach by comparing its predicted "
        "prices with the actual stock prices."
    )


    selected_stock = st.selectbox(
        "Choose a Company",
        STOCKS,
        format_func=lambda x: STOCK_NAMES[x]
    )


    stock_results = comparison[
        comparison["Stock"] == selected_stock
    ].sort_values("RMSE")


    st.markdown("---")


    # BEST MODEL

    best = best_models[
        best_models["Stock"] == selected_stock
    ]


    if not best.empty:

        best_model = best.iloc[0]["Model"]

        st.markdown(

            f"""
            <div class="success-box">

            <h3>🏆 Best Performing Approach</h3>

            <h2>{best_model}</h2>

            This approach achieved the best performance
            according to the selected test evaluation metric.

            </div>
            """,

            unsafe_allow_html=True

        )


    st.markdown("---")


    # PERFORMANCE METRICS

    st.subheader("Compare Prediction Accuracy")

    st.write(
        "Lower error values generally indicate more accurate "
        "predictions. Higher R² values indicate a better "
        "statistical fit."
    )


    display_results = stock_results[
        ["Model", "MAE", "RMSE", "MAPE (%)", "R2"]
    ].copy()

    st.dataframe(
        display_results.round(4),
        use_container_width=True,
        hide_index=True
    )


    # RMSE GRAPH

    st.markdown("---")

    st.subheader("Which Approach Made Smaller Prediction Errors?")

    rmse_fig = px.bar(

        stock_results,

        x="Model",

        y="RMSE",

        color="Model",

        text="RMSE",

        title="Prediction Error Comparison"

    )

    rmse_fig.update_layout(
        template="plotly_white",
        showlegend=False,
        xaxis_title="Prediction Approach",
        yaxis_title="RMSE (USD)"
    )

    st.plotly_chart(
        rmse_fig,
        use_container_width=True
    )


    # EXPLAIN THE METRICS

    st.markdown("---")

    st.subheader("Understanding the Results")


    with st.expander("What do these performance measurements mean?"):

        st.markdown(
            """
            **1. MAE – Average Prediction Error**

            Shows how far the predicted prices were from
            the actual prices on average, in USD.

            Smaller values are better.

            ---

            **2. RMSE – Prediction Error with Greater Penalty
            for Large Errors**

            Similar to MAE, but gives more importance to
            larger prediction errors.

            Smaller values are better.

            ---

            **3. MAPE – Average Percentage Error**

            Shows the average prediction error as a percentage.

            Smaller values are better.

            ---

            **4. R² – Model Fit**

            Indicates how well the model explains the
            variation in actual prices.

            Higher values generally indicate a better fit.
            Negative values are possible and indicate
            poor performance relative to predicting
            the mean of the actual values.
            """
        )


    st.info(
        "A model performing best on one company does not "
        "necessarily mean it will perform best on another."
    )


# ==================================================
# 11. RISK AND RETURNS
# ==================================================

elif page == "💡 Risk & Returns":

    st.header("Understanding Historical Risk and Returns")

    st.write(
        "Stock prices do not move in a straight line. "
        "This section helps you understand how frequently "
        "prices change and how much they fluctuate."
    )


    selected_stock = st.selectbox(
        "Choose a Company",
        STOCKS,
        format_func=lambda x: STOCK_NAMES[x]
    )


    stock_returns = returns[selected_stock].dropna()


    # CALCULATIONS

    annualised_volatility = (
        stock_returns.std() * np.sqrt(252) * 100
    )

    average_daily_return = (
        stock_returns.mean() * 100
    )


    # SUMMARY CARDS

    col1, col2 = st.columns(2)


    col1.metric(
        "Historical Price Fluctuation",
        f"{annualised_volatility:.2f}%"
    )


    col2.metric(
        "Average Daily Price Change",
        f"{average_daily_return:.3f}%"
    )


    st.markdown("---")


    # EXPLANATION

    st.subheader("What Does Historical Risk Mean?")

    st.markdown("""
    Historical volatility measures how much a stock's
    daily returns have fluctuated over time.

    - Higher volatility means larger historical fluctuations.
    - Lower volatility means smaller historical fluctuations.

    This is a measure of past price behaviour, not a
    guarantee of future performance.
    """)


    # VOLATILITY CHART

    st.subheader("How Has Price Fluctuation Changed Over Time?")


    stock_features = features[
        features["Stock"] == selected_stock
    ]


    volatility_fig = px.line(

        stock_features,

        x="Date",

        y="Volatility_10",

        title="Historical 10-Day Price Fluctuation",

        labels={
            "Volatility_10": "10-Day Volatility",
            "Date": "Date"
        }

    )


    volatility_fig.update_layout(
        template="plotly_white"
    )


    st.plotly_chart(
        volatility_fig,
        use_container_width=True
    )


    st.info(
        "This section provides historical financial analysis "
        "for academic and educational purposes. "
        "It does not provide investment recommendations."
    )


# ==================================================
# 12. RESEARCH RESULTS
# ==================================================

elif page == "🎓 Research Results":

    st.header("Our Research Findings")

    st.write(
        "This section presents the main findings from "
        "our comparative stock price prediction research."
    )


    # FINAL MODEL COMPARISON

    st.subheader("Overall Model Performance")


    st.dataframe(

        comparison.sort_values(
            ["Stock", "RMSE"]
        ).round(4),

        use_container_width=True,

        hide_index=True

    )


    st.markdown("---")


    # BEST MODELS

    st.subheader("Best Performing Approaches")


    st.dataframe(

        best_models,

        use_container_width=True,

        hide_index=True

    )


    st.markdown("---")


    # VALIDATION

    st.subheader("Validation Results")


    st.write(
        "Validation results help us assess model performance "
        "during the model development process."
    )


    st.dataframe(

        validation.round(4),

        use_container_width=True,

        hide_index=True

    )


    st.markdown("---")


    # KEY FINDINGS

    st.subheader("Key Research Observations")


    st.markdown("""

    **1. Simpler approaches can perform very well.**

    The Naïve Persistence Baseline achieved the lowest
    test RMSE across all three stocks.

    **2. GRU showed strong performance among the
    advanced prediction models.**

    It was the strongest of the six machine learning
    and deep learning models in the test comparison.

    **3. Different stocks produced different results.**

    The performance of prediction approaches varied
    depending on the company being analysed.

    **4. More complex models do not always perform better.**

    Increasing model complexity does not automatically
    improve prediction accuracy.

    """)


    st.markdown("---")


    # FINAL CONCLUSION

    st.subheader("What Did We Learn?")


    st.success(
        """
        Our research demonstrates that stock price prediction
        performance depends on the selected model, stock and
        experimental setup.

        The results highlight the importance of comparing
        advanced machine learning and deep learning models
        against a simple baseline rather than assuming that
        more complex models will always perform better.
        """
    )


    st.info(
        "These findings are limited to the selected companies, "
        "historical dataset, features, evaluation period and "
        "experimental methodology."
    )


# ==================================================
# 13. FOOTER
# ==================================================

st.markdown("---")

st.markdown(

    """
    <div style="text-align:center;">

    <h4>Stock Insights | Final Year Research Project</h4>

    <p>
    BSc Financial Engineering<br>
    Department of Mathematics<br>
    Faculty of Science<br>
    University of Colombo
    </p>

    <p>
    <strong>Bimalya Herath & Tharuka Hewawasam</strong>
    </p>

    <p style="color:gray;font-size:12px;">
    Academic research project | 2026
    </p>

    </div>
    """,

    unsafe_allow_html=True

)
