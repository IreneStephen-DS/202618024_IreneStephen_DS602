import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Medical Insurance Statistical Dashboard",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📊 Medical Insurance Statistical Modeling Dashboard")

st.markdown(
    """
    **M.Sc. Data Science — Semester 1**

    **Lab-4: Applied Statistical Modeling & Interactive Web Dashboard**

    This dashboard performs:
    - Exploratory Data Analysis
    - Hypothesis Testing
    - Multiple Linear Regression
    - Residual Diagnostics
    - Multicollinearity Analysis
    - Live Medical Insurance Cost Prediction
    """
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    file_path = "data/insurance.csv"

    try:
        df = pd.read_csv(file_path)

    except Exception:
        df = pd.read_csv(file_path, sep="\t")

    # Remove unnecessary spaces from column names
    df.columns = df.columns.str.strip()

    return df


df = load_data()


# ============================================================
# DATA CLEANING
# ============================================================

# Standardize column names
df.columns = df.columns.str.lower().str.strip()

# Remove duplicate rows
df = df.drop_duplicates()

# Convert numerical columns
numerical_columns = [
    "age",
    "bmi",
    "children",
    "charges"
]

for col in numerical_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# Remove missing values
df = df.dropna()


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "age",
    "sex",
    "bmi",
    "children",
    "smoker",
    "region",
    "charges"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        f"""
        The following required columns are missing:

        {missing_columns}

        Your dataset contains:

        {list(df.columns)}
        """
    )

    st.stop()


# ============================================================
# CREATE ADDITIONAL FEATURES
# ============================================================

df["bmi_squared"] = df["bmi"] ** 2

df["smoker_binary"] = (
    df["smoker"]
    .astype(str)
    .str.lower()
    .map({"yes": 1, "no": 0})
)

df["smoker_binary"] = df["smoker_binary"].fillna(0)

df["smoker_bmi"] = (
    df["smoker_binary"] * df["bmi"]
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Dashboard Controls")

st.sidebar.write(
    f"Total observations: **{len(df)}**"
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "📊 Data Exploration",
        "🧪 Hypothesis Testing Lab",
        "🤖 Prediction & Diagnostics"
    ]
)


# ============================================================
# TAB 1 — DATA EXPLORATION
# ============================================================

with tab1:

    st.header("📊 Data Exploration")

    st.subheader("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Rows",
            df.shape[0]
        )

    with col2:
        st.metric(
            "Columns",
            df.shape[1]
        )

    with col3:
        st.metric(
            "Average Charges",
            f"${df['charges'].mean():,.2f}"
        )

    with col4:
        st.metric(
            "Median Charges",
            f"${df['charges'].median():,.2f}"
        )


    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    st.subheader("Interactive Filters")

    col1, col2 = st.columns(2)

    with col1:

        age_min = int(df["age"].min())
        age_max = int(df["age"].max())

        age_range = st.slider(
            "Age Range",
            min_value=age_min,
            max_value=age_max,
            value=(age_min, age_max)
        )

    with col2:

        selected_regions = st.multiselect(
            "Select Region",
            options=sorted(df["region"].unique()),
            default=sorted(df["region"].unique())
        )


    filtered_df = df[
        (df["age"] >= age_range[0]) &
        (df["age"] <= age_range[1]) &
        (df["region"].isin(selected_regions))
    ]


    st.write(
        f"Filtered observations: **{len(filtered_df)}**"
    )


    # --------------------------------------------------------
    # DATA PREVIEW
    # --------------------------------------------------------

    st.subheader("Filtered Dataset")

    st.dataframe(
        filtered_df[
            [
                "age",
                "sex",
                "bmi",
                "children",
                "smoker",
                "region",
                "charges"
            ]
        ],
        use_container_width=True
    )


    # --------------------------------------------------------
    # DESCRIPTIVE STATISTICS
    # --------------------------------------------------------

    st.subheader("Descriptive Statistics")

    descriptive = pd.DataFrame({
        "Mean": filtered_df[
            numerical_columns
        ].mean(),

        "Median": filtered_df[
            numerical_columns
        ].median(),

        "Standard Deviation": filtered_df[
            numerical_columns
        ].std(),

        "IQR": (
            filtered_df[numerical_columns].quantile(0.75)
            -
            filtered_df[numerical_columns].quantile(0.25)
        ),

        "Skewness": filtered_df[
            numerical_columns
        ].skew(),

        "Kurtosis": filtered_df[
            numerical_columns
        ].kurt()
    })

    st.dataframe(
        descriptive.round(4),
        use_container_width=True
    )


    # --------------------------------------------------------
    # HISTOGRAM
    # --------------------------------------------------------

    st.subheader("Distribution Plot")

    selected_numeric = st.selectbox(
        "Select numerical variable",
        numerical_columns
    )

    fig_hist = px.histogram(
        filtered_df,
        x=selected_numeric,
        marginal="box",
        nbins=30,
        title=f"Distribution of {selected_numeric}"
    )

    st.plotly_chart(
        fig_hist,
        use_container_width=True
    )


    # --------------------------------------------------------
    # SCATTER PLOT
    # --------------------------------------------------------

    st.subheader("Bivariate Analysis")

    col1, col2 = st.columns(2)

    with col1:

        x_variable = st.selectbox(
            "X Variable",
            [
                "age",
                "bmi",
                "children"
            ]
        )

    with col2:

        y_variable = st.selectbox(
            "Y Variable",
            [
                "charges",
                "bmi",
                "age"
            ]
        )


    fig_scatter = px.scatter(
        filtered_df,
        x=x_variable,
        y=y_variable,
        color="smoker",
        hover_data=[
            "age",
            "bmi",
            "charges",
            "region"
        ],
        trendline="ols",
        title=f"{y_variable} vs {x_variable}"
    )

    st.plotly_chart(
        fig_scatter,
        use_container_width=True
    )


    # --------------------------------------------------------
    # CORRELATION MATRIX
    # --------------------------------------------------------

    st.subheader("Correlation Matrix")

    correlation = filtered_df[
        numerical_columns
    ].corr()

    fig_corr = px.imshow(
        correlation,
        text_auto=True,
        title="Correlation Matrix"
    )

    st.plotly_chart(
        fig_corr,
        use_container_width=True
    )


# ============================================================
# TAB 2 — HYPOTHESIS TESTING
# ============================================================

with tab2:

    st.header("🧪 Hypothesis Testing Lab")

    st.markdown(
        """
        Significance level:

        **α = 0.05**
        """
    )


    # ========================================================
    # HYPOTHESIS TEST 1
    # ========================================================

    st.subheader(
        "Hypothesis Test 1 — Two Group Comparison"
    )

    st.markdown(
        """
        **Example:** Compare medical charges between smokers
        and non-smokers.

        **H₀:** There is no significant difference in the
        selected numerical variable between the two groups.

        **H₁:** There is a significant difference between the
        two groups.
        """
    )


    group_variable = st.selectbox(
        "Select grouping variable",
        [
            "smoker",
            "sex"
        ],
        key="group_variable"
    )


    numeric_variable = st.selectbox(
        "Select numerical variable",
        [
            "charges",
            "age",
            "bmi",
            "children"
        ],
        key="numeric_variable"
    )


    groups = df[group_variable].dropna().unique()


    if len(groups) == 2:

        group1 = df[
            df[group_variable] == groups[0]
        ][numeric_variable].dropna()

        group2 = df[
            df[group_variable] == groups[1]
        ][numeric_variable].dropna()


        st.write(
            f"Group 1: **{groups[0]}**"
        )

        st.write(
            f"Group 2: **{groups[1]}**"
        )


        # ----------------------------------------------------
        # SHAPIRO-WILK TEST
        # ----------------------------------------------------

        st.write("### Normality Test — Shapiro-Wilk")


        # Shapiro can become problematic for very large samples.
        # Use a sample of 5000 maximum.

        sample1 = group1.sample(
            min(len(group1), 5000),
            random_state=42
        )

        sample2 = group2.sample(
            min(len(group2), 5000),
            random_state=42
        )


        shapiro1 = stats.shapiro(
            sample1
        )

        shapiro2 = stats.shapiro(
            sample2
        )


        col1, col2 = st.columns(2)

        with col1:

            st.write(
                f"**{groups[0]}**"
            )

            st.write(
                f"Statistic: {shapiro1.statistic:.4f}"
            )

            st.write(
                f"p-value: {shapiro1.pvalue:.6f}"
            )


        with col2:

            st.write(
                f"**{groups[1]}**"
            )

            st.write(
                f"Statistic: {shapiro2.statistic:.4f}"
            )

            st.write(
                f"p-value: {shapiro2.pvalue:.6f}"
            )


        normal1 = shapiro1.pvalue > 0.05
        normal2 = shapiro2.pvalue > 0.05


        # ----------------------------------------------------
        # LEVENE TEST
        # ----------------------------------------------------

        st.write(
            "### Equal Variance Test — Levene's Test"
        )


        levene_test = stats.levene(
            group1,
            group2
        )


        st.write(
            f"Statistic: **{levene_test.statistic:.4f}**"
        )

        st.write(
            f"p-value: **{levene_test.pvalue:.6f}**"
        )


        equal_variance = (
            levene_test.pvalue > 0.05
        )


        # ----------------------------------------------------
        # SELECT STATISTICAL TEST
        # ----------------------------------------------------

        st.write("### Final Statistical Test")


        if normal1 and normal2:

            if equal_variance:

                test_result = stats.ttest_ind(
                    group1,
                    group2,
                    equal_var=True
                )

                test_name = (
                    "Independent Two-Sample t-test"
                )

            else:

                test_result = stats.ttest_ind(
                    group1,
                    group2,
                    equal_var=False
                )

                test_name = (
                    "Welch's Two-Sample t-test"
                )

        else:

            test_result = stats.mannwhitneyu(
                group1,
                group2,
                alternative="two-sided"
            )

            test_name = (
                "Mann-Whitney U test"
            )


        st.write(
            f"**Test used:** {test_name}"
        )

        st.write(
            f"Test statistic: **{test_result.statistic:.4f}**"
        )

        st.write(
            f"p-value: **{test_result.pvalue:.6f}**"
        )


        alpha = 0.05


        if test_result.pvalue < alpha:

            st.error(
                """
                **Reject H₀**

                There is statistically significant evidence
                that the two groups differ.
                """
            )

        else:

            st.success(
                """
                **Fail to Reject H₀**

                There is insufficient statistical evidence
                to conclude that the two groups differ.
                """
            )


    # ========================================================
    # HYPOTHESIS TEST 2 — ANOVA
    # ========================================================

    st.divider()

    st.subheader(
        "Hypothesis Test 2 — One-Way ANOVA"
    )


    st.markdown(
        """
        We test whether the mean medical charges differ
        across the geographic regions.

        **H₀:** All group means are equal.

        **H₁:** At least one group mean is different.
        """
    )


    anova_variable = st.selectbox(
        "Select numerical variable for ANOVA",
        [
            "charges",
            "age",
            "bmi"
        ]
    )


    anova_groups = []

    for region in df["region"].unique():

        values = df[
            df["region"] == region
        ][anova_variable].dropna()

        anova_groups.append(values)


    anova_result = stats.f_oneway(
        *anova_groups
    )


    st.write(
        f"F-statistic: **{anova_result.statistic:.4f}**"
    )

    st.write(
        f"p-value: **{anova_result.pvalue:.6f}**"
    )


    if anova_result.pvalue < 0.05:

        st.error(
            """
            **Reject H₀**

            There is statistically significant evidence
            that at least one regional mean differs.
            """
        )

    else:

        st.success(
            """
            **Fail to Reject H₀**

            There is insufficient evidence that the
            regional means differ.
            """
        )


    # ANOVA visualization

    fig_box = px.box(
        df,
        x="region",
        y=anova_variable,
        color="region",
        title=f"{anova_variable} by Region"
    )

    st.plotly_chart(
        fig_box,
        use_container_width=True
    )


# ============================================================
# TAB 3 — REGRESSION + PREDICTION + DIAGNOSTICS
# ============================================================

with tab3:

    st.header(
        "🤖 Live Prediction & Diagnostics"
    )


    # ========================================================
    # PREPARE DATA FOR REGRESSION
    # ========================================================

    model_df = df.copy()


    # Dummy variables

    model_df = pd.get_dummies(
        model_df,
        columns=[
            "sex",
            "smoker",
            "region"
        ],
        drop_first=True,
        dtype=int
    )


    # Predictor variables

    predictor_columns = [
        "age",
        "bmi",
        "children",
        "bmi_squared",
        "smoker_binary",
        "smoker_bmi"
    ]


    dummy_columns = [
        col
        for col in model_df.columns
        if (
            col.startswith("sex_")
            or
            col.startswith("smoker_")
            or
            col.startswith("region_")
        )
        and col != "smoker_binary"
    ]


    predictor_columns.extend(
        dummy_columns
    )


    # Remove duplicate columns

    predictor_columns = list(
        dict.fromkeys(
            predictor_columns
        )
    )


    X = model_df[
        predictor_columns
    ].astype(float)


    y = model_df[
        "charges"
    ].astype(float)


    # Add intercept

    X_with_constant = sm.add_constant(
        X
    )


    # ========================================================
    # OLS MODEL
    # ========================================================

    model = sm.OLS(
        y,
        X_with_constant
    ).fit()


    # ========================================================
    # MODEL SUMMARY
    # ========================================================

    st.subheader(
        "Multiple Linear Regression — OLS"
    )


    st.markdown(
        """
        The fitted model predicts medical insurance charges
        using demographic, health, smoking and regional
        variables.
        """
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "R²",
            f"{model.rsquared:.4f}"
        )


    with col2:

        st.metric(
            "Adjusted R²",
            f"{model.rsquared_adj:.4f}"
        )


    with col3:

        st.metric(
            "F-statistic",
            f"{model.fvalue:.2f}"
        )


    with col4:

        st.metric(
            "Model p-value",
            f"{model.f_pvalue:.6f}"
        )


    # ========================================================
    # COEFFICIENT TABLE
    # ========================================================

    st.subheader(
        "Regression Coefficients"
    )


    coefficient_table = pd.DataFrame({
        "Coefficient": model.params,
        "Std Error": model.bse,
        "t-statistic": model.tvalues,
        "p-value": model.pvalues,
        "CI Lower 95%": model.conf_int()[0],
        "CI Upper 95%": model.conf_int()[1]
    })


    st.dataframe(
        coefficient_table.round(4),
        use_container_width=True
    )


    # ========================================================
    # MODEL INTERPRETATION
    # ========================================================

    st.subheader(
        "Important Coefficients"
    )


    significant_coefficients = (
        coefficient_table[
            coefficient_table["p-value"] < 0.05
        ]
        .sort_values("p-value")
    )


    if len(significant_coefficients) > 0:

        st.write(
            "Statistically significant predictors "
            "at α = 0.05:"
        )

        st.dataframe(
            significant_coefficients.round(4),
            use_container_width=True
        )

    else:

        st.write(
            "No statistically significant predictors "
            "were found at α = 0.05."
        )


    # ========================================================
    # RESIDUALS
    # ========================================================

    residuals = model.resid
    fitted_values = model.fittedvalues


    # ========================================================
    # RESIDUAL VS FITTED
    # ========================================================

    st.subheader(
        "Residuals vs Fitted Values"
    )


    fig_residual = px.scatter(
        x=fitted_values,
        y=residuals,
        labels={
            "x": "Fitted Values",
            "y": "Residuals"
        },
        title="Residuals vs Fitted Values"
    )


    fig_residual.add_hline(
        y=0,
        line_dash="dash"
    )


    st.plotly_chart(
        fig_residual,
        use_container_width=True
    )


    st.markdown(
        """
        **Interpretation:**

        Ideally, residuals should be randomly scattered
        around zero with approximately constant spread.

        A funnel-shaped pattern may indicate
        heteroscedasticity.
        """
    )


    # ========================================================
    # Q-Q PLOT
    # ========================================================

    st.subheader(
        "Normal Q-Q Plot"
    )


    fig_qq = sm.qqplot(
        residuals,
        line="45",
        fit=True
    )


    st.pyplot(
        fig_qq.figure
    )

    plt.close(
        fig_qq.figure
    )


    # ========================================================
    # JARQUE-BERA TEST
    # ========================================================

    st.subheader(
        "Normality of Residuals — Jarque-Bera Test"
    )


    jb_test = stats.jarque_bera(
        residuals
    )


    st.write(
        f"Jarque-Bera Statistic: "
        f"**{jb_test.statistic:.4f}**"
    )

    st.write(
        f"p-value: "
        f"**{jb_test.pvalue:.6f}**"
    )


    if jb_test.pvalue < 0.05:

        st.warning(
            """
            **Reject normality assumption.**

            The residuals show statistically significant
            evidence of departure from normality.
            """
        )

    else:

        st.success(
            """
            **Fail to Reject normality assumption.**

            There is insufficient evidence to conclude
            that the residuals are non-normal.
            """
        )


    # ========================================================
    # BREUSCH-PAGAN TEST
    # ========================================================

    st.subheader(
        "Homoscedasticity — Breusch-Pagan Test"
    )


    bp_test = het_breuschpagan(
        residuals,
        X_with_constant
    )


    bp_labels = [
        "LM Statistic",
        "LM p-value",
        "F Statistic",
        "F p-value"
    ]


    bp_results = pd.Series(
        bp_test,
        index=bp_labels
    )


    st.dataframe(
        bp_results.to_frame(
            "Value"
        ).round(6)
    )


    if bp_test[1] < 0.05:

        st.warning(
            """
            **Reject H₀ of homoscedasticity.**

            Evidence suggests that the residual variance
            is not constant.
            """
        )

    else:

        st.success(
            """
            **Fail to Reject H₀ of homoscedasticity.**

            There is insufficient evidence of
            heteroscedasticity.
            """
        )


    # ========================================================
    # VIF
    # ========================================================

    st.subheader(
        "Multicollinearity — Variance Inflation Factor"
    )


    vif_data = pd.DataFrame()

    vif_data["Variable"] = X.columns

    vif_data["VIF"] = [
        variance_inflation_factor(
            X.values,
            i
        )
        for i in range(X.shape[1])
    ]


    st.dataframe(
        vif_data.round(4),
        use_container_width=True
    )


    st.markdown(
        """
        **VIF interpretation:**

        - VIF ≈ 1 → No multicollinearity
        - VIF between 1 and 5 → Generally acceptable
        - VIF > 5 → Potential multicollinearity
        - VIF > 10 → Serious multicollinearity
        """
    )


    # ========================================================
    # LIVE PREDICTION
    # ========================================================

    st.divider()

    st.header(
        "💰 Live Medical Insurance Cost Prediction"
    )


    st.write(
        """
        Enter patient characteristics below to generate
        a predicted insurance charge and a 95% prediction
        interval.
        """
    )


    col1, col2 = st.columns(2)


    with col1:

        input_age = st.number_input(
            "Age",
            min_value=int(df["age"].min()),
            max_value=int(df["age"].max()),
            value=int(df["age"].median())
        )


        input_bmi = st.number_input(
            "BMI",
            min_value=float(df["bmi"].min()),
            max_value=float(df["bmi"].max()),
            value=float(df["bmi"].median())
        )


        input_children = st.number_input(
            "Number of Children",
            min_value=int(df["children"].min()),
            max_value=int(df["children"].max()),
            value=int(df["children"].median())
        )


    with col2:

        input_sex = st.selectbox(
            "Sex",
            sorted(df["sex"].unique())
        )


        input_smoker = st.selectbox(
            "Smoker",
            sorted(df["smoker"].unique())
        )


        input_region = st.selectbox(
            "Region",
            sorted(df["region"].unique())
        )


    # ========================================================
    # CREATE PREDICTION DATA
    # ========================================================

    input_data = pd.DataFrame(
        {
            "age": [input_age],
            "bmi": [input_bmi],
            "children": [input_children],
            "bmi_squared": [
                input_bmi ** 2
            ],
            "smoker_binary": [
                1 if str(input_smoker).lower() == "yes"
                else 0
            ],
            "smoker_bmi": [
                (
                    1 if str(input_smoker).lower() == "yes"
                    else 0
                ) * input_bmi
            ]
        }
    )


    # Add dummy columns

    for col in dummy_columns:

        input_data[col] = 0


    # Match categorical dummy columns

    for col in dummy_columns:

        if col.startswith("sex_"):

            category = col.replace(
                "sex_",
                "",
                1
            )

            if str(input_sex) == category:

                input_data[col] = 1


        elif col.startswith("smoker_"):

            category = col.replace(
                "smoker_",
                "",
                1
            )

            if str(input_smoker) == category:

                input_data[col] = 1


        elif col.startswith("region_"):

            category = col.replace(
                "region_",
                "",
                1
            )

            if str(input_region) == category:

                input_data[col] = 1


    # Ensure same column order

    input_data = input_data[
        predictor_columns
    ]


    input_data = input_data.astype(float)


    input_with_constant = sm.add_constant(
        input_data,
        has_constant="add"
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    prediction = model.get_prediction(
        input_with_constant
    )


    prediction_summary = prediction.summary_frame(
        alpha=0.05
    )


    predicted_charge = prediction_summary[
        "mean"
    ].iloc[0]


    mean_ci_lower = prediction_summary[
        "mean_ci_lower"
    ].iloc[0]


    mean_ci_upper = prediction_summary[
        "mean_ci_upper"
    ].iloc[0]


    prediction_lower = prediction_summary[
        "obs_ci_lower"
    ].iloc[0]


    prediction_upper = prediction_summary[
        "obs_ci_upper"
    ].iloc[0]


    # ========================================================
    # DISPLAY PREDICTION
    # ========================================================

    st.subheader(
        "Prediction Result"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Predicted Charge",
            f"${predicted_charge:,.2f}"
        )


    with col2:

        st.metric(
            "95% Prediction Lower",
            f"${prediction_lower:,.2f}"
        )


    with col3:

        st.metric(
            "95% Prediction Upper",
            f"${prediction_upper:,.2f}"
        )


    st.info(
        f"""
        **Predicted medical insurance charge:**

        ${predicted_charge:,.2f}

        **95% Prediction Interval:**

        ${prediction_lower:,.2f}
        to
        ${prediction_upper:,.2f}

        The prediction interval represents the range in
        which an individual future observation is expected
        to fall with approximately 95% confidence under
        the fitted model assumptions.
        """
    )


    # ========================================================
    # CONFIDENCE INTERVAL FOR MEAN RESPONSE
    # ========================================================

    st.write(
        "### 95% Confidence Interval for Mean Prediction"
    )


    st.write(
        f"""
        Lower bound: **${mean_ci_lower:,.2f}**

        Upper bound: **${mean_ci_upper:,.2f}**
        """
    )


    # ========================================================
    # MODEL SUMMARY OPTION
    # ========================================================

    with st.expander(
        "View Complete OLS Model Summary"
    ):

        st.text(
            model.summary()
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "M.Sc. Data Science — Statistical Modeling with Python | Lab-4"
)