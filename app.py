import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Stock Price Prediction | Final Year Research",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# BASIC STYLING
# ============================================================

st.markdown("""
<style>

    .main {
        background-color: #ffffff;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1 {
        font-weight: 650;
        color: #1f2937;
        letter-spacing: -0.5px;
    }

    h2 {
        color: #1f2937;
        margin-top: 1.5rem;
    }

    h3 {
        color: #374151;
    }

    p {
        color: #374151;
        line-height: 1.65;
    }

    .research-box {
        background-color: #f5f7fa;
        border-left: 4px solid #374151;
        padding: 1.2rem 1.4rem;
        margin: 1rem 0 1.5rem 0;
        border-radius: 4px;
    }

    .finding-box {
        background-color: #f3f7f4;
        border-left: 4px solid #4b6b57;
        padding: 1.3rem 1.5rem;
        margin: 1rem 0 1.5rem 0;
        border-radius: 4px;
    }

    .warning-box {
        background-color: #faf8f2;
        border-left: 4px solid #8a7650;
        padding: 1.2rem 1.4rem;
        margin: 1rem 0 1.5rem 0;
        border-radius: 4px;
    }

    .student-name {
        font-size: 1rem;
        color: #4b5563;
        line-height: 1.8;
    }

    .section-description {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #6b7280;
    }

    .sidebar-title {
        font-size: 1.15rem;
        font-weight: 600;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA
# ============================================================

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
        "The research data files could not be found. "
        "Please make sure the 'research_outputs' folder "
        "contains all required CSV files."
    )

    st.stop()


# ============================================================
# GENERAL SETTINGS
# ============================================================

STOCKS = ["AAPL", "MSFT", "TSLA"]

STOCK_NAMES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "TSLA": "Tesla"
}


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    '<div class="sidebar-title">Stock Price Prediction Research</div>',
    unsafe_allow_html=True
)

st.sidebar.caption(
    "Final Year Research Project"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "The Data",
        "What We Tested",
        "Results",
        "Stock Behaviour",
        "Research Conclusion"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "BSc Financial Engineering\n"
    "Department of Mathematics\n"
    "Faculty of Science\n"
    "University of Colombo"
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.title("Stock Price Prediction for Data-Driven Decision Support")

    st.markdown(
        '<div class="section-description">'
        'Final Year Research Project'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="student-name">
        <strong>Bimalya Herath</strong> (24SFE037)<br>
        <strong>Tharuka Hewawasam</strong> (24SFE031)<br><br>

        BSc Financial Engineering<br>
        Department of Mathematics<br>
        Faculty of Science<br>
        University of Colombo
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # --------------------------------------------------------
    # WHY
    # --------------------------------------------------------

    st.header("Why did we do this research?")

    st.write(
        """
        Stock prices change continuously and can be difficult to
        predict accurately. Historical market data may contain
        patterns that can be used to estimate future prices.
        """
    )

    st.write(
        """
        Machine learning and deep learning provide different
        approaches for identifying these patterns. However,
        a more complicated model does not automatically mean
        a more accurate prediction.
        """
    )

    st.markdown(
        """
        <div class="research-box">

        <strong>Our research question</strong><br><br>

        Can machine learning and deep learning models improve
        stock price prediction compared with a simple baseline
        approach?

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # WHAT
    # --------------------------------------------------------

    st.header("What did we investigate?")

    st.write(
        """
        We compared different prediction approaches using
        historical stock data for three large companies:
        """
    )

    cols = st.columns(3)

    for col, stock in zip(cols, STOCKS):

        with col:

            st.subheader(STOCK_NAMES[stock])

            st.write(
                f"Stock ticker: **{stock}**"
            )

    # --------------------------------------------------------
    # HOW
    # --------------------------------------------------------

    st.header("How did we do it?")

    steps = [
        (
            "1. Collect the data",
            "Historical stock market data was collected for "
            "Apple, Microsoft and Tesla."
        ),
        (
            "2. Prepare the data",
            "The data was cleaned and transformed into variables "
            "that could be used by prediction models."
        ),
        (
            "3. Train prediction models",
            "Several machine learning and deep learning approaches "
            "were trained using the prepared data."
        ),
        (
            "4. Test the models",
            "The models generated predictions for data that was "
            "not used to train them."
        ),
        (
            "5. Compare the results",
            "Predicted prices were compared with actual prices "
            "using several performance measures."
        )
    ]

    for title, description in steps:

        st.markdown(f"**{title}**")

        st.write(description)

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    st.header("What did we find?")

    st.markdown(
        """
        <div class="finding-box">

        <strong>Main finding</strong><br><br>

        The most important result of the research was that
        <strong>greater model complexity did not necessarily
        produce better prediction accuracy.</strong>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(
        """
        The Naïve Persistence Baseline achieved the lowest
        test RMSE across the three stocks.

        Among the six machine learning and deep learning
        approaches tested, GRU was the strongest performer
        in the test comparison.
        """
    )

    st.write(
        """
        Model performance also varied between companies,
        showing that one prediction approach does not
        necessarily work best for every stock.
        """
    )

    # --------------------------------------------------------
    # IMPORTANT NOTE
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="warning-box">

        <strong>Important</strong><br><br>

        These results describe performance on the selected
        stocks, dataset, features, evaluation period and
        experimental setup. They should not be interpreted
        as guaranteed future stock prices or investment advice.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# THE DATA
# ============================================================

elif page == "The Data":

    st.title("The Data")

    st.markdown(
        '<div class="section-description">'
        'What information did we use for the research?'
        '</div>',
        unsafe_allow_html=True
    )

    st.header("Which stocks were analysed?")

    stock_table = pd.DataFrame({
        "Company": [
            "Apple",
            "Microsoft",
            "Tesla"
        ],
        "Ticker": [
            "AAPL",
            "MSFT",
            "TSLA"
        ]
    })

    st.dataframe(
        stock_table,
        use_container_width=True,
        hide_index=True
    )

    st.header("What information did we use?")

    st.write(
        """
        The research uses historical stock market information.
        The original price data was transformed into additional
        variables that could help the models identify patterns.
        """
    )

    feature_explanations = pd.DataFrame({
        "Variable": [
            "Closing Price",
            "Daily Return",
            "5-Day Moving Average",
            "10-Day Moving Average",
            "20-Day Moving Average",
            "10-Day Volatility"
        ],
        "Simple explanation": [
            "The stock's closing price for a trading day.",
            "The percentage change in price from the previous trading day.",
            "The average closing price over the previous 5 trading days.",
            "The average closing price over the previous 10 trading days.",
            "The average closing price over the previous 20 trading days.",
            "A measure of how much the stock price has been fluctuating recently."
        ]
    })

    st.dataframe(
        feature_explanations,
        use_container_width=True,
        hide_index=True
    )

    st.header("Why were these variables used?")

    st.write(
        """
        Stock prices contain information about recent price
        movements and fluctuations. Moving averages can help
        represent recent price trends, while daily returns
        and volatility describe how the price has been changing.
        """
    )


# ============================================================
# WHAT WE TESTED
# ============================================================

elif page == "What We Tested":

    st.title("What We Tested")

    st.markdown(
        '<div class="section-description">'
        'The prediction approaches used in the research'
        '</div>',
        unsafe_allow_html=True
    )

    st.header("Prediction approaches")

    models = {
        "Naïve Persistence Baseline":
            "Uses the most recent observed price as the prediction. "
            "This provides a simple benchmark for judging whether "
            "more advanced models actually improve prediction accuracy.",

        "Random Forest":
            "Uses multiple decision trees and combines their "
            "predictions.",

        "XGBoost":
            "Uses a sequence of decision trees designed to improve "
            "prediction performance.",

        "SVR":
            "Support Vector Regression, a machine learning method "
            "that estimates a continuous value such as stock price.",

        "LSTM":
            "A deep learning model designed to learn patterns "
            "from sequential data.",

        "GRU":
            "A recurrent deep learning model that learns patterns "
            "from sequences while using a simpler structure than LSTM.",

        "Transformer":
            "A deep learning architecture that uses attention "
            "mechanisms to identify relationships within sequential data."
    }

    for model, explanation in models.items():

        st.subheader(model)

        st.write(explanation)

        st.markdown("---")

    st.header("How were the models compared?")

    st.write(
        """
        Each model produced stock price predictions for the
        test period. These predictions were then compared with
        the actual prices.
        """
    )

    metrics = pd.DataFrame({
        "Measure": [
            "MAE",
            "RMSE",
            "MAPE",
            "R²"
        ],
        "What it tells us": [
            "The average size of the prediction error.",
            "The size of prediction errors, with larger errors receiving greater weight.",
            "The average prediction error expressed as a percentage.",
            "How well the model explains variation in the actual prices."
        ],
        "Preferred result": [
            "Lower",
            "Lower",
            "Lower",
            "Higher"
        ]
    })

    st.dataframe(
        metrics,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# RESULTS
# ============================================================

elif page == "Results":

    st.title("Results")

    st.markdown(
        '<div class="section-description">'
        'What did the models actually produce?'
        '</div>',
        unsafe_allow_html=True
    )

    selected_stock = st.selectbox(
        "Select a company",
        STOCKS,
        format_func=lambda x: STOCK_NAMES[x]
    )

    stock_results = comparison[
        comparison["Stock"] == selected_stock
    ].sort_values("RMSE")

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    best = best_models[
        best_models["Stock"] == selected_stock
    ]

    if not best.empty:

        best_model = best.iloc[0]["Model"]

        st.markdown(
            f"""
            <div class="finding-box">

            <strong>Best test-performing approach for
            {STOCK_NAMES[selected_stock]}</strong><br><br>

            <span style="font-size:1.3rem;">
            {best_model}
            </span>

            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    st.header("Prediction performance")

    st.write(
        """
        The table below shows how closely each approach
        predicted the actual stock price.
        """
    )

    result_table = stock_results[
        ["Model", "MAE", "RMSE", "MAPE (%)", "R2"]
    ].copy()

    result_table = result_table.round(4)

    st.dataframe(
        result_table,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    st.header("Prediction error")

    st.write(
        """
        RMSE is one of the main measures used in this research.
        A smaller RMSE means that the model's predictions were,
        on average, closer to the actual prices, while giving
        greater weight to larger errors.
        """
    )

    rmse_fig = px.bar(
        stock_results,
        x="Model",
        y="RMSE",
        text="RMSE",
        title=f"RMSE for {STOCK_NAMES[selected_stock]}"
    )

    rmse_fig.update_layout(
        template="plotly_white",
        showlegend=False,
        xaxis_title="Prediction Approach",
        yaxis_title="RMSE"
    )

    st.plotly_chart(
        rmse_fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # ACTUAL VS PREDICTED
    # --------------------------------------------------------

    st.header("Actual price compared with predictions")

    stock_predictions = predictions[
        predictions["Stock"] == selected_stock
    ].sort_values("Date")

    available_models = [
        model for model in [
            "Naive Baseline",
            "Random Forest",
            "XGBoost",
            "SVR",
            "LSTM",
            "GRU",
            "Transformer"
        ]
        if model in stock_predictions.columns
    ]

    selected_models = st.multiselect(
        "Select prediction approaches",
        available_models,
        default=available_models
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

        fig.add_trace(
            go.Scatter(
                x=stock_predictions["Date"],
                y=stock_predictions[model],
                mode="lines",
                name=model
            )
        )

    fig.update_layout(
        title=f"{STOCK_NAMES[selected_stock]}: Actual Price vs Predictions",
        xaxis_title="Date",
        yaxis_title="Closing Price (USD)",
        hovermode="x unified",
        template="plotly_white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# STOCK BEHAVIOUR
# ============================================================

elif page == "Stock Behaviour":

    st.title("Stock Behaviour")

    st.markdown(
        '<div class="section-description">'
        'Understanding how the three stocks changed over time'
        '</div>',
        unsafe_allow_html=True
    )

    selected_stock = st.selectbox(
        "Select a company",
        STOCKS,
        format_func=lambda x: STOCK_NAMES[x]
    )

    prices = historical[selected_stock].dropna()

    latest = prices.iloc[-1]

    previous = prices.iloc[-2]

    change = latest - previous

    percentage = (change / previous) * 100

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Latest closing price",
        f"${latest:.2f}"
    )

    col2.metric(
        "Change from previous day",
        f"${change:.2f}",
        f"{percentage:.2f}%"
    )

    col3.metric(
        "Trading days in dataset",
        f"{len(prices):,}"
    )

    st.header("Historical price")

    price_fig = px.line(
        x=prices.index,
        y=prices.values,
        title=f"{STOCK_NAMES[selected_stock]} Historical Closing Price",
        labels={
            "x": "Date",
            "y": "Closing Price (USD)"
        }
    )

    price_fig.update_layout(
        template="plotly_white"
    )

    st.plotly_chart(
        price_fig,
        use_container_width=True
    )

    st.header("Daily price changes")

    stock_returns = returns[selected_stock].dropna()

    return_fig = px.line(
        x=stock_returns.index,
        y=stock_returns.values * 100,
        title="Daily Percentage Price Changes",
        labels={
            "x": "Date",
            "y": "Daily Change (%)"
        }
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

    st.header("Historical volatility")

    stock_features = features[
        features["Stock"] == selected_stock
    ]

    volatility_fig = px.line(
        stock_features,
        x="Date",
        y="Volatility_10",
        title="10-Day Historical Volatility",
        labels={
            "Date": "Date",
            "Volatility_10": "Volatility"
        }
    )

    volatility_fig.update_layout(
        template="plotly_white"
    )

    st.plotly_chart(
        volatility_fig,
        use_container_width=True
    )

    st.markdown(
        """
        <div class="warning-box">

        <strong>How should volatility be understood?</strong><br><br>

        Volatility describes how much prices have fluctuated
        historically. Higher volatility means larger price
        movements. It does not tell us exactly what will happen
        to the stock in the future.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# RESEARCH CONCLUSION
# ============================================================

elif page == "Research Conclusion":

    st.title("Research Conclusion")

    st.markdown(
        '<div class="section-description">'
        'What do the results tell us?'
        '</div>',
        unsafe_allow_html=True
    )

    st.header("Main findings")

    st.markdown(
        """
        **1. The simple baseline was difficult to beat.**

        The Naïve Persistence Baseline achieved the lowest
        test RMSE across all three stocks.

        **2. GRU was the strongest advanced approach.**

        Among the six machine learning and deep learning
        approaches, GRU produced the strongest overall
        test performance in our comparison.

        **3. Model performance depended on the stock.**

        The same approach did not necessarily perform
        equally well across Apple, Microsoft and Tesla.

        **4. More complex does not necessarily mean better.**

        The results do not support the assumption that
        increasingly complex models will automatically
        produce more accurate stock price predictions.
        """
    )

    st.header("What did we learn?")

    st.write(
        """
        The main lesson from this research is that model
        selection should be based on actual performance
        rather than complexity alone.
        """
    )

    st.write(
        """
        A sophisticated deep learning model may be useful,
        but it should always be evaluated against an appropriate
        baseline. In our experiment, the simple persistence
        approach provided a very strong benchmark.
        """
    )

    st.header("Why is this important?")

    st.write(
        """
        The purpose of this project was not simply to build
        the most complicated prediction model.

        The purpose was to investigate whether different
        modelling approaches could provide more accurate
        stock price predictions and to determine which
        approaches performed best under our experimental
        conditions.
        """
    )

    st.markdown(
        """
        <div class="finding-box">

        <strong>Final conclusion</strong><br><br>

        Our findings show that stock price prediction remains
        challenging. Advanced machine learning and deep learning
        models can provide useful predictions, but greater
        complexity does not guarantee improved accuracy.

        The results therefore support the importance of
        benchmarking advanced models against simple baseline
        approaches.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.header("Scope of the findings")

    st.write(
        """
        These conclusions apply specifically to the selected
        stocks, historical dataset, engineered features,
        prediction methods, evaluation period and experimental
        methodology used in this research.

        The predictions should not be interpreted as guaranteed
        future stock prices or as investment recommendations.
        """)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Stock Price Prediction for Data-Driven Decision Support | "
    "Final Year Research Project | University of Colombo"
)
