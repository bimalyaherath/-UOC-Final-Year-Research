import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Stock Market Prediction Research",
    page_icon="",
    layout="wide"
)


# ==========================================================
# 2. DESIGN
# ==========================================================

st.markdown("""
<style>

    .main {
        background-color: #F8FAFC;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1 {
        color: #17365D;
        font-weight: 700;
    }

    h2, h3 {
        color: #24476B;
    }

    p {
        line-height: 1.7;
    }

    .intro-box {
        background-color: #EDF3F8;
        padding: 24px;
        border-radius: 8px;
        border-left: 4px solid #315D83;
        margin: 15px 0 25px 0;
    }

    .finding-box {
        background-color: #FFFFFF;
        padding: 20px;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        margin-bottom: 15px;
    }

    .research-box {
        background-color: #FFFFFF;
        padding: 22px;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        margin-bottom: 15px;
    }

    .muted {
        color: #64748B;
        font-size: 14px;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        padding: 18px;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
    }

    div[data-testid="stMetricLabel"] {
        color: #64748B;
    }

    hr {
        border-color: #E2E8F0;
    }

</style>
""", unsafe_allow_html=True)


# ==========================================================
# 3. RESEARCH INFORMATION
# ==========================================================

PROJECT_TITLE = (
    "Stock Market Price Prediction "
    "for Data-Driven Decision Support"
)

STOCKS = ["AAPL", "MSFT", "TSLA"]

STOCK_NAMES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "TSLA": "Tesla"
}

MODELS = [
    "Naive Baseline",
    "Random Forest",
    "XGBoost",
    "SVR",
    "LSTM",
    "GRU",
    "Transformer"
]

MODEL_DESCRIPTIONS = {

    "Naive Baseline":
    "Uses the most recently observed price as the next "
    "price estimate. It provides a simple benchmark "
    "against which more advanced approaches are compared.",

    "Random Forest":
    "Combines multiple decision trees to estimate "
    "stock prices using patterns in the available data.",

    "XGBoost":
    "Uses a sequence of decision trees to improve "
    "prediction performance.",

    "SVR":
    "Uses Support Vector Regression to estimate "
    "stock prices based on relationships in the data.",

    "LSTM":
    "A type of recurrent neural network designed to "
    "learn patterns from sequential information.",

    "GRU":
    "A recurrent neural network that learns relationships "
    "in sequential data using a simplified gating structure.",

    "Transformer":
    "Uses attention mechanisms to learn relationships "
    "between different parts of sequential data."

}


# ==========================================================
# 4. LOAD RESEARCH DATA
# ==========================================================

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
        "Research data files are missing. "
        "Please check the research_outputs folder."
    )

    st.stop()


# ==========================================================
# 5. HELPER FUNCTIONS
# ==========================================================

def get_stock_name(stock):

    return STOCK_NAMES.get(stock, stock)


def get_best_model(stock):

    result = best_models[
        best_models["Stock"] == stock
    ]

    if result.empty:
        return None

    return result.iloc[0]["Model"]


def get_model_results(stock):

    result = comparison[
        comparison["Stock"] == stock
    ].copy()

    return result.sort_values("RMSE")


def format_date(date):

    return pd.to_datetime(date).strftime("%d %B %Y")


# ==========================================================
# 6. SIDEBAR
# ==========================================================

st.sidebar.title("Research Dashboard")

st.sidebar.caption(
    "Stock Market Price Prediction"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(

    "Navigation",

    [
        "Home",
        "Explore the Market",
        "Our Predictions",
        "How Accurate Were We?",
        "What Did We Discover?",
        "About This Research"
    ]

)

st.sidebar.markdown("---")

st.sidebar.caption(
    "BSc Financial Engineering"
)

st.sidebar.caption(
    "University of Colombo"
)


# ==========================================================
# 7. MAIN HEADER
# ==========================================================

st.title("Stock Market Prediction Research")

st.markdown(
    """
    A comparative study of machine learning and deep learning
    approaches for stock price prediction.
    """
)

st.markdown("---")


# ==========================================================
# 8. HOME PAGE
# ==========================================================

if page == "Home":

    st.header("Understanding Stock Price Predictions")

    st.markdown(
        """
        <div class="intro-box">

        <h3>Why did we conduct this research?</h3>

        Stock prices change continuously, making it difficult
        to estimate how they may move in the future.

        Machine learning and deep learning offer different
        ways of analysing historical market information
        and identifying patterns that may help estimate
        stock prices.

        However, using a more advanced model does not
        automatically mean obtaining better predictions.

        <br><br>

        <strong>
        Our research investigates how accurately different
        prediction approaches estimate stock prices and
        whether advanced models outperform a simple baseline.
        </strong>

        </div>
        """,
        unsafe_allow_html=True
    )


    # RESEARCH QUESTION

    st.subheader("Our Main Research Question")

    st.markdown(
        """
        Can machine learning and deep learning models
        predict stock prices more accurately than a
        simple baseline approach?
        """
    )


    st.markdown("---")


    # COMPANIES

    st.subheader("Companies Included in Our Research")

    cols = st.columns(3)

    for col, stock in zip(cols, STOCKS):

        price = historical[stock].dropna()

        latest = price.iloc[-1]

        with col:

            st.markdown(
                f"""
                <div class="research-box">

                <h3>{get_stock_name(stock)}</h3>

                <p class="muted">
                Stock Symbol: {stock}
                </p>

                <h2>${latest:.2f}</h2>

                <p class="muted">
                Latest available closing price
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )


    st.markdown("---")


    # RESEARCH PROCESS

    st.subheader("How Did We Conduct the Research?")

    steps = [

        (
            "01. Data Collection",
            "Collected historical stock market information "
            "for the three selected companies."
        ),

        (
            "02. Data Preparation",
            "Prepared the data and developed additional "
            "features for prediction."
        ),

        (
            "03. Model Development",
            "Developed and evaluated seven different "
            "prediction approaches."
        ),

        (
            "04. Price Prediction",
            "Generated estimated stock prices using "
            "the developed models."
        ),

        (
            "05. Performance Evaluation",
            "Compared the estimated prices with actual "
            "prices using statistical evaluation metrics."
        ),

        (
            "06. Research Findings",
            "Identified the strongest-performing approaches "
            "and interpreted the results."
        )

    ]

    for title, description in steps:

        st.markdown(
            f"""
            <div class="finding-box">

            <strong>{title}</strong>

            <p>{description}</p>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown("---")


    # RESEARCH SUMMARY

    st.subheader("Research at a Glance")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Companies Analysed",
        "3"
    )

    c2.metric(
        "Prediction Approaches",
        "7"
    )

    c3.metric(
        "Research Area",
        "Predictive Modelling"
    )


    st.info(
        "Explore the sections of this dashboard to understand "
        "the data, predictions, evaluation results and "
        "conclusions of our research."
    )


# ==========================================================
# 9. EXPLORE THE MARKET
# ==========================================================

elif page == "Explore the Market":

    st.header("Explore Historical Stock Prices")

    st.write(
        "Select a company to explore how its stock price "
        "has changed throughout the research period."
    )


    stock = st.selectbox(
        "Select a company",
        STOCKS,
        format_func=get_stock_name
    )


    price = historical[stock].dropna()

    stock_returns = returns[stock].dropna()


    # PRICE SUMMARY

    latest = price.iloc[-1]

    previous = price.iloc[-2]

    change = latest - previous

    percentage = change / previous * 100


    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Latest Closing Price",
        f"${latest:.2f}"
    )

    c2.metric(
        "Previous Day's Change",
        f"${change:.2f}",
        f"{percentage:.2f}%"
    )

    c3.metric(
        "Trading Days Available",
        f"{len(price):,}"
    )


    st.markdown("---")


    # HISTORICAL PRICE

    st.subheader(
        f"How Has {get_stock_name(stock)}'s Price Changed?"
    )

    st.write(
        "This graph shows the recorded closing price "
        "of the selected company over the research period."
    )


    fig = px.line(

        x=price.index,

        y=price.values,

        labels={
            "x": "Date",
            "y": "Closing Price (USD)"
        }

    )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # DAILY CHANGES

    st.markdown("---")

    st.subheader("Daily Stock Price Changes")

    st.write(
        "Stock prices can increase or decrease each trading day. "
        "This graph shows the percentage change between "
        "consecutive trading days."
    )


    fig = px.line(

        x=stock_returns.index,

        y=stock_returns.values * 100,

        labels={
            "x": "Date",
            "y": "Daily Price Change (%)"
        }

    )

    fig.add_hline(
        y=0,
        line_dash="dash"
    )

    fig.update_layout(
        template="plotly_white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    with st.expander("How should I understand this graph?"):

        st.write(
            """
            Positive values indicate that the stock price
            increased compared with the previous trading day.

            Negative values indicate that the stock price
            decreased.

            Larger movements indicate greater daily
            price fluctuations.
            """
        )


# ==========================================================
# 10. OUR PREDICTIONS
# ==========================================================

elif page == "Our Predictions":

    st.header("What Did Our Models Predict?")

    st.write(
        """
        We developed seven prediction approaches and used
        them to estimate stock prices.

        This section allows you to compare their estimates
        with the actual stock market prices recorded during
        the test period.
        """
    )


    stock = st.selectbox(
        "Select a company",
        STOCKS,
        format_func=get_stock_name
    )


    data = predictions[
        predictions["Stock"] == stock
    ].copy()

    data = data.sort_values("Date")


    available_models = [

        model for model in MODELS

        if model in data.columns

    ]


    selected_models = st.multiselect(

        "Select prediction approaches to display",

        available_models,

        default=available_models

    )


    # ACTUAL VS PREDICTED

    fig = go.Figure()


    fig.add_trace(

        go.Scatter(

            x=data["Date"],

            y=data["Actual"],

            mode="lines",

            name="Actual Market Price",

            line=dict(width=4)

        )

    )


    for model in selected_models:

        fig.add_trace(

            go.Scatter(

                x=data["Date"],

                y=data[model],

                mode="lines",

                name=model

            )

        )


    fig.update_layout(

        title=f"{get_stock_name(stock)}: Actual and Estimated Prices",

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


    # EXPLANATION

    st.subheader("Understanding the Graph")

    st.markdown(
        """
        **Actual Market Price**

        The actual closing price recorded in the stock market.

        **Predicted Price**

        The estimated closing price generated by a
        prediction model.

        **What should you look for?**

        The closer a model's prediction line is to the
        actual price line, the more closely its estimates
        followed the recorded market prices.
        """
    )


    st.markdown("---")


    # MODEL EXPLANATIONS

    st.subheader("How Do the Different Approaches Work?")


    with st.expander("Learn about the seven prediction approaches"):

        for model in available_models:

            st.markdown(f"**{model}**")

            st.write(
                MODEL_DESCRIPTIONS[model]
            )

            st.markdown("---")


    st.warning(
        "These are estimates produced during the historical "
        "test period. They should not be interpreted as "
        "guaranteed future stock prices."
    )


# ==========================================================
# 11. HOW ACCURATE WERE WE?
# ==========================================================

elif page == "How Accurate Were We?":

    st.header("Evaluating Prediction Accuracy")

    st.write(
        """
        A prediction is useful only if it is reasonably close
        to the actual stock price.

        We therefore compared the predictions from all
        seven approaches against the actual prices to
        identify their relative performance.
        """
    )


    stock = st.selectbox(
        "Select a company",
        STOCKS,
        format_func=get_stock_name
    )


    results = get_model_results(stock)


    st.markdown("---")


    # BEST MODEL

    best = get_best_model(stock)


    if best is not None:

        st.markdown(
            f"""
            <div class="intro-box">

            <h3>Best Performing Approach</h3>

            <h2>{best}</h2>

            This approach achieved the lowest test RMSE
            among the approaches evaluated for
            {get_stock_name(stock)}.

            </div>
            """,
            unsafe_allow_html=True
        )


    # PERFORMANCE TABLE

    st.subheader("Comparison of All Prediction Approaches")

    st.write(
        "The table shows how closely each approach "
        "estimated the actual stock prices."
    )


    columns = [
        "Model",
        "MAE",
        "RMSE",
        "MAPE (%)",
        "R2"
    ]


    st.dataframe(

        results[columns].round(4),

        use_container_width=True,

        hide_index=True

    )


    st.markdown("---")


    # RMSE CHART

    st.subheader("Comparing Prediction Errors")

    st.write(
        """
        RMSE measures the difference between predicted
        and actual prices, while giving greater importance
        to larger errors.

        A smaller RMSE indicates better performance
        under this evaluation measure.
        """
    )


    fig = px.bar(

        results.sort_values("RMSE"),

        x="Model",

        y="RMSE",

        color="Model",

        text="RMSE"

    )


    fig.update_layout(

        template="plotly_white",

        showlegend=False,

        xaxis_title="Prediction Approach",

        yaxis_title="RMSE (USD)"

    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # METRIC EXPLANATION

    st.markdown("---")

    st.subheader("What Do These Measurements Mean?")


    explanations = {

        "MAE – Average Prediction Error":

        """
        Measures the average absolute difference between
        predicted prices and actual prices.

        It is expressed in USD. Smaller values indicate
        more accurate predictions.
        """,

        "RMSE – Prediction Error":

        """
        Measures prediction errors while giving greater
        importance to larger errors.

        It is also expressed in USD. Smaller values
        indicate better performance.
        """,

        "MAPE – Percentage Prediction Error":

        """
        Measures the average prediction error as a
        percentage of actual prices.

        Smaller values generally indicate better performance.
        """,

        "R² – Model Fit":

        """
        Measures how well the model explains variation
        in actual prices.

        Higher values generally indicate a better fit.
        Negative values are possible and indicate poor
        performance relative to a mean-based prediction.
        """

    }


    for title, explanation in explanations.items():

        with st.expander(title):

            st.write(explanation)


# ==========================================================
# 12. WHAT DID WE DISCOVER?
# ==========================================================

elif page == "What Did We Discover?":

    st.header("The Results of Our Research")

    st.write(
        """
        After evaluating seven prediction approaches
        across three companies, we identified several
        important findings.
        """
    )


    # FINDING 1

    st.markdown(
        """
        <div class="finding-box">

        <h3>1. The Simple Baseline Outperformed
        the Advanced Models</h3>

        The Naïve Persistence Baseline achieved the
        lowest test RMSE across all three stocks.

        This demonstrates that more complicated
        prediction approaches do not necessarily
        produce more accurate results.

        </div>
        """,
        unsafe_allow_html=True
    )


    # FINDING 2

    st.markdown(
        """
        <div class="finding-box">

        <h3>2. GRU Was the Strongest Advanced Approach</h3>

        Among the six machine learning and deep learning
        models, GRU demonstrated the strongest performance
        in the test comparison.

        </div>
        """,
        unsafe_allow_html=True
    )


    # FINDING 3

    st.markdown(
        """
        <div class="finding-box">

        <h3>3. Model Performance Varied Between Companies</h3>

        The prediction results were not identical
        across Apple, Microsoft and Tesla.

        This highlights the importance of evaluating
        models separately for different stocks.

        </div>
        """,
        unsafe_allow_html=True
    )


    # FINDING 4

    st.markdown(
        """
        <div class="finding-box">

        <h3>4. Greater Complexity Did Not Guarantee
        Better Accuracy</h3>

        The results show that using more advanced
        machine learning and deep learning architectures
        does not automatically improve stock price
        prediction performance.

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown("---")


    # COMPLETE RESULTS

    st.subheader("Complete Model Comparison Results")

    st.write(
        """
        The following table presents the numerical
        evaluation results for all three companies.
        """
    )


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


    # VALIDATION

    st.markdown("---")

    st.subheader("Model Validation Results")

    st.write(
        """
        These results provide additional information
        about model performance during the validation
        stage of the research.
        """
    )


    st.dataframe(
        validation.round(4),
        use_container_width=True,
        hide_index=True
    )


    # FINAL CONCLUSION

    st.markdown("---")

    st.subheader("Overall Research Conclusion")


    st.markdown(
        """
        Our research demonstrates that stock price
        prediction performance depends on the selected
        approach, the company being analysed and the
        experimental setup.

        Although advanced machine learning and deep
        learning models can identify patterns in
        historical market data, their increased
        complexity does not necessarily translate
        into better prediction accuracy.

        The strong performance of the simple baseline
        highlights the importance of comparing advanced
        prediction models against appropriate benchmark
        approaches.
        """
    )


    st.info(
        "These findings are specific to the selected "
        "companies, historical dataset, features, "
        "evaluation period and research methodology."
    )


# ==========================================================
# 13. ABOUT THIS RESEARCH
# ==========================================================

elif page == "About This Research":

    st.header("About Our Research Project")


    # PROJECT TITLE

    st.subheader("Project Title")

    st.markdown(
        f"### {PROJECT_TITLE}"
    )


    st.markdown("---")


    # WHY

    st.subheader("Why Was This Research Conducted?")

    st.write(
        """
        Stock price prediction is a challenging task
        because market prices change over time and
        may contain complex patterns.

        Different machine learning and deep learning
        models offer different ways of analysing
        historical market information.

        However, it is important to determine whether
        these advanced approaches actually provide
        better predictions than simpler alternatives.

        This research was therefore conducted to
        systematically compare their prediction
        performance.
        """
    )


    # WHAT

    st.markdown("---")

    st.subheader("What Did We Investigate?")

    st.write(
        """
        The research focuses on the comparative
        evaluation of seven prediction approaches
        across three selected companies.

        The primary objective is to assess their
        ability to estimate stock prices using
        historical market data.
        """
    )


    # HOW

    st.markdown("---")

    st.subheader("How Was the Research Conducted?")

    methodology = [

        (
            "Data Collection",
            "Historical stock market data for Apple, "
            "Microsoft and Tesla."
        ),

        (
            "Data Preparation",
            "Preparation of historical observations "
            "and development of relevant features."
        ),

        (
            "Model Development",
            "Implementation of machine learning and "
            "deep learning prediction approaches."
        ),

        (
            "Prediction",
            "Generation of estimated stock prices "
            "for the evaluation period."
        ),

        (
            "Evaluation",
            "Comparison using MAE, RMSE, MAPE and R²."
        )

    ]


    for title, description in methodology:

        st.markdown(f"**{title}**")

        st.write(description)


    # WHEN

    st.markdown("---")

    st.subheader("Research Period")

    st.write(
        f"""
        The available historical dataset covers the period
        from {format_date(historical.index.min())}
        to {format_date(historical.index.max())}.
        """
    )


    st.caption(
        "This represents the available historical data period, "
        "not necessarily the full duration of the research project."
    )


    # WHO

    st.markdown("---")

    st.subheader("Research Team")


    st.markdown(
        """
        <div class="research-box">

        <h3>Bimalya Herath</h3>

        Student ID: 24sfe037

        </div>

        <div class="research-box">

        <h3>Tharuka Hewawasam</h3>

        Student ID: 24sfe031

        </div>
        """,
        unsafe_allow_html=True
    )


    # UNIVERSITY

    st.markdown("---")

    st.subheader("Academic Affiliation")

    st.markdown(
        """
        **Bachelor of Science in Financial Engineering**

        Department of Mathematics

        Faculty of Science

        University of Colombo

        Sri Lanka
        """
    )


    # OUTPUT

    st.markdown("---")

    st.subheader("What Did This Research Produce?")

    st.write(
        """
        The final outputs of this research include:

        - A prepared historical stock market dataset.
        - An engineered dataset containing additional
          predictive features.
        - Seven developed stock price prediction approaches.
        - Historical test-period price predictions.
        - Comparative model evaluation results.
        - Validation results.
        - A comparative analysis of model performance.
        - A research dashboard presenting the findings.

        These outputs provide a structured comparison
        of the prediction approaches investigated
        in this study.
        """
    )


# ==========================================================
# 14. FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;">

    <p><strong>Stock Market Price Prediction for
    Data-Driven Decision Support</strong></p>

    <p>
    Final Year Research Project<br>
    BSc Financial Engineering<br>
    Department of Mathematics<br>
    Faculty of Science<br>
    University of Colombo
    </p>

    <p class="muted">
    Bimalya Herath (24sfe037) |
    Tharuka Hewawasam (24sfe031)
    </p>

    </div>
    """,
    unsafe_allow_html=True
)
