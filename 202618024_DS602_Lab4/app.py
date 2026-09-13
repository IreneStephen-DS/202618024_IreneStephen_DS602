import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera

# ==========================================
# LOAD DATASET
# ==========================================
@st.cache_data
def load_data():
    return pd.read_csv("data/insurance.csv")


df = load_data()


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Medical Insurance Statistical Dashboard",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Medical Insurance Analytics & Diagnostic Dashboard")


# Navigation Tabs
tab1, tab2, tab3 = st.tabs([
    "📊 Data Exploration",
    "🧪 Hypothesis Testing Lab",
    "📈 Live Prediction & Diagnostics"
])


# ==========================================
# TAB 1: EDA
# ==========================================
with tab1:

    st.header("📊 Exploratory Data Analysis")

    # ======================================
    # DATASET OVERVIEW
    # ======================================

    st.subheader("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    # Number of rows
    col1.metric(
        "Rows",
        f"{df.shape[0]:,}"
    )

    # Number of columns
    col2.metric(
        "Columns",
        f"{df.shape[1]}"
    )

    # Average charges
    col3.metric(
        "Average Charges",
        f"${df['charges'].mean():,.2f}"
    )

    # Median charges
    col4.metric(
        "Median Charges",
        f"${df['charges'].median():,.2f}"
    )

    st.divider()
# ======================================
    # DATASET PREVIEW
    # ======================================

    st.markdown("### 1. Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    # ======================================
    # DESCRIPTIVE STATISTICS
    # ======================================

    st.subheader("2. Descriptive Metrics")

    st.write(
        "The following table presents measures of central tendency, "
        "dispersion, skewness, and kurtosis for all numerical features."
    )

    # Select numerical columns
    numerical_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    # Create descriptive statistics dataframe
    descriptive_stats = pd.DataFrame(index=numerical_columns)

    # Mean
    descriptive_stats["Mean"] = df[numerical_columns].mean()

    # Median
    descriptive_stats["Median"] = df[numerical_columns].median()

    # Standard Deviation
    descriptive_stats["Standard Deviation"] = df[numerical_columns].std()

    # IQR = Q3 - Q1
    descriptive_stats["IQR"] = (
        df[numerical_columns].quantile(0.75)
        - df[numerical_columns].quantile(0.25)
    )

    # Skewness
    descriptive_stats["Skewness"] = df[numerical_columns].skew()

    # Kurtosis
    descriptive_stats["Kurtosis"] = df[numerical_columns].kurtosis()

    # Display table
    st.dataframe(
        descriptive_stats.style.format("{:.3f}"),
        use_container_width=True
    )

    st.divider()


    # ======================================
    # VISUAL EXPLORATION
    # ======================================

    st.subheader("3. Visual Exploration")

    st.write(
        "Explore distributions, bivariate relationships, and "
        "correlations among numerical variables."
    )


    # ======================================
    # DISTRIBUTION PLOT
    # ======================================

    st.markdown("### 3.1. Distribution Plot")
    st.write(
        "Visualize the distribution of a selected numerical variable ")
    distribution_variable = st.selectbox(
        "Select Distribution Variable",
        numerical_columns,
        index=(
            numerical_columns.index("charges")
            if "charges" in numerical_columns
            else 0
        )
    )

    # Reduced graph size
    fig, ax = plt.subplots(figsize=(5, 3))

    # Histogram
    sns.histplot(
        data=df,
        x=distribution_variable,
        kde=True,
        bins=30,
        color="steelblue",
        edgecolor="black",
        alpha=0.7,
        ax=ax
    )

    ax.set_title(
        f"Distribution of {distribution_variable.title()}",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel(
        distribution_variable.title(),
        fontsize=11
    )

    ax.set_ylabel(
        "Frequency",
        fontsize=11
    )

    # Do not stretch graph to full browser width
    st.pyplot(fig, use_container_width=False)
    st.divider()


    # ======================================
    # BIVARIATE SCATTER PLOT
    # ======================================

    st.markdown("### 3.2. Bivariate Scatter Plot")
    st.write(
        "Examine the relationship between two numerical variables ")
    scatter_col1, scatter_col2 = st.columns(2)

    with scatter_col1:

        x_variable = st.selectbox(
            "Select X-axis Variable",
            numerical_columns,
            index=(
                numerical_columns.index("age")
                if "age" in numerical_columns
                else 0
            ),
            key="scatter_x"
        )

    with scatter_col2:

        y_variable = st.selectbox(
            "Select Y-axis Variable",
            numerical_columns,
            index=(
                numerical_columns.index("charges")
                if "charges" in numerical_columns
                else 0
            ),
            key="scatter_y"
        )


    # Calculate correlation between selected variables
    correlation = df[x_variable].corr(
        df[y_variable]
    )

    # Reduced graph size
    fig, ax = plt.subplots(figsize=(6, 3.5))

    sns.scatterplot(
        data=df,
        x=x_variable,
        y=y_variable,
        color="darkorange",
        alpha=0.65,
        s=60,
        ax=ax
    )

    # Regression line
    sns.regplot(
        data=df,
        x=x_variable,
        y=y_variable,
        scatter=False,
        color="red",
        line_kws={"linewidth": 2},
        ax=ax
    )

    ax.set_title(
        f"{x_variable.title()} vs {y_variable.title()}",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel(
        x_variable.title(),
        fontsize=11
    )

    ax.set_ylabel(
        y_variable.title(),
        fontsize=11
    )

    # Do not stretch graph to full browser width
    st.pyplot(fig, use_container_width=False)

    st.metric(
        f"Pearson Correlation: {x_variable} vs {y_variable}",
        f"{correlation:.3f}"
    )

    st.divider()


    # ======================================
    # CORRELATION MATRIX
    # ======================================

    st.markdown("### 3.3. Correlation Matrix")
    st.write(
        "Visualize the pairwise correlations between numerical variables ")
    st.write(
        "The correlation matrix shows the strength and direction "
        "of linear relationships between numerical variables."
    )

    correlation_matrix = df[numerical_columns].corr()

    # Reduced heatmap size
    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        square=True,
        annot_kws={"size": 9},
        ax=ax
    )

    ax.set_title(
        "Correlation Matrix of Numerical Features",
        fontsize=12,
        fontweight="bold"
    )

    # Do not stretch heatmap to full browser width
    st.pyplot(fig, use_container_width=False)

    st.divider()

# ==========================================
# TAB 2: HYPOTHESIS TESTING LAB
# ==========================================
with tab2:

    st.header("🧪 Hypothesis Testing Lab")

    # --------------------------------------
    # Hypothesis Test 1: Smokers vs Non-Smokers
    # --------------------------------------

    st.subheader(
        "Hypothesis Test 1: Smokers vs Non-Smokers"
    )

    st.caption(
        "Compare medical charges between smokers and non-smokers."
    )

    # ======================================
    # GROUP DEFINITIONS
    # ======================================

    group1 = "yes"
    group2 = "no"

    group1_label = "Smokers"
    group2_label = "Non-Smokers"

    continuous_metric = "charges"


    # ======================================
    # HYPOTHESES
    # ======================================

    st.markdown("### 1. Hypotheses")

    st.write(
        "**H₀ (Null Hypothesis):** "
        "There is no significant difference in medical charges "
        "between smokers and non-smokers."
    )

    st.write(
        "**H₁ (Alternative Hypothesis):** "
        "There is a significant difference in medical charges "
        "between smokers and non-smokers."
    )

    st.info(
        "This is a two-sided test because we are testing whether "
        "medical charges are different between the two groups."
    )


    # ======================================
    # EXTRACT THE TWO GROUPS
    # ======================================

    smokers = df[
        df["smoker"] == group1
    ]["charges"].dropna()

    non_smokers = df[
        df["smoker"] == group2
    ]["charges"].dropna()


    # ======================================
    # GROUP SUMMARY
    # ======================================

    st.markdown("### 2. Group Summary")

    g1, g2 = st.columns(2)

    with g1:

        st.write("**Smokers**")

        st.metric(
            "Sample Size",
            f"{len(smokers):,}"
        )

        st.write(
            f"Mean Medical Charges: **${smokers.mean():,.2f}**"
        )

    with g2:

        st.write("**Non-Smokers**")

        st.metric(
            "Sample Size",
            f"{len(non_smokers):,}"
        )

        st.write(
            f"Mean Medical Charges: **${non_smokers.mean():,.2f}**"
        )


    st.divider()


    # ======================================
    # SHAPIRO-WILK NORMALITY TEST
    # ======================================

    st.markdown(
        "### 3. Normality Check — Shapiro-Wilk Test"
    )

    st.write(
        "The Shapiro-Wilk test checks whether medical charges "
        "within each group are consistent with a normal distribution."
    )

    shapiro_smokers = stats.shapiro(smokers)
    shapiro_non_smokers = stats.shapiro(non_smokers)

    s1, s2 = st.columns(2)

    with s1:

        st.write("**Smokers**")

        st.metric(
            "W Statistic",
            f"{shapiro_smokers.statistic:.4f}"
        )

        (
    f"{shapiro_smokers.pvalue:.6f}"
    if shapiro_smokers.pvalue >= 0.000001
    else "< 0.000001"
)

        if shapiro_smokers.pvalue < 0.05:

            st.error(
                "p < 0.05 → Reject normality H₀. "
                "The smoker charges are not normally distributed."
            )

        else:

            st.success(
                "p ≥ 0.05 → Fail to Reject normality H₀. "
                "No significant evidence of non-normality."
            )

    with s2:

        st.write("**Non-Smokers**")

        st.metric(
            "W Statistic",
            f"{shapiro_non_smokers.statistic:.4f}"
        )

        (
    f"{shapiro_non_smokers.pvalue:.6f}"
    if shapiro_non_smokers.pvalue >= 0.000001
    else "< 0.000001"
)

        if shapiro_non_smokers.pvalue < 0.05:

            st.error(
                "p < 0.05 → Reject normality H₀. "
                "The non-smoker charges are not normally distributed."
            )

        else:

            st.success(
                "p ≥ 0.05 → Fail to Reject normality H₀. "
                "No significant evidence of non-normality."
            )


    # Determine whether BOTH groups are normal
    both_groups_normal = (
        shapiro_smokers.pvalue >= 0.05
        and
        shapiro_non_smokers.pvalue >= 0.05
    )


    st.divider()


    # ======================================
    # LEVENE'S EQUAL VARIANCE TEST
    # ======================================

    st.markdown(
        "### 4. Equal Variance Check — Levene's Test"
    )

    st.write(
        "Levene's test checks whether the variances of "
        "smokers and non-smokers are statistically equal."
    )

    levene_result = stats.levene(
        smokers,
        non_smokers,
        center="median"
    )

    l1, l2 = st.columns(2)

    with l1:

        st.metric(
            "Levene Statistic",
            f"{levene_result.statistic:.4f}"
        )

    with l2:

        st.metric(
            "p-value",
            f"{levene_result.pvalue:.6f}"
        )

    if levene_result.pvalue < 0.05:

        st.error(
            "p < 0.05 → Reject H₀. "
            "The variances of the two groups are significantly different."
        )

    else:

        st.success(
            "p ≥ 0.05 → Fail to Reject H₀. "
            "No significant evidence that the group variances differ."
        )


    st.divider()


    # ======================================
    # TEST SELECTION
    # ======================================

    st.markdown(
        "### 5. Statistical Test Selection"
    )

    if both_groups_normal:

        st.success(
            "Both groups satisfy the normality check. "
            "Therefore, an Independent Two-Sample t-test is selected."
        )

        # Two-sample t-test
        # If variances are unequal, Welch's correction is used.
        ttest_result = stats.ttest_ind(
            smokers,
            non_smokers,
            equal_var=(
                levene_result.pvalue >= 0.05
            )
        )

        selected_test = "Independent Two-Sample t-test"

        test_statistic = ttest_result.statistic

        p_value = ttest_result.pvalue

        if levene_result.pvalue < 0.05:

            st.caption(
                "Levene's test indicates unequal variances, "
                "so Welch's correction is used for the t-test."
            )

        else:

            st.caption(
                "Levene's test indicates equal variances, "
                "so the standard independent two-sample t-test is used."
            )

    else:

        st.warning(
            "At least one group does not satisfy the normality check. "
            "Therefore, the Mann-Whitney U test is selected."
        )

        # Mann-Whitney U test
        mann_whitney_result = stats.mannwhitneyu(
            smokers,
            non_smokers,
            alternative="two-sided"
        )

        selected_test = "Mann-Whitney U Test"

        test_statistic = mann_whitney_result.statistic

        p_value = mann_whitney_result.pvalue


    st.divider()


    # ======================================
    # TEST RESULTS
    # ======================================

    st.markdown(
        "### 6. Test Results"
    )

    r1, r2, r3 = st.columns(3)

    with r1:

        st.metric(
            "Selected Test",
            selected_test
        )

    with r2:

        st.metric(
            "Test Statistic",
            f"{test_statistic:.4f}"
        )

    with r3:

        st.metric(
            "p-value",
            f"{p_value:.6f}"
        )


    # ======================================
    # FINAL DECISION
    # ======================================

    alpha = 0.05

    st.markdown(
        f"### 7. Final Conclusion at α = {alpha}"
    )

    if p_value < alpha:

        st.error(
            "## Reject H₀"
        )

        st.write(
            f"The p-value ({p_value:.6f}) is less than "
            f"α = {alpha}."
        )

        st.write(
            "**Conclusion:** There is statistically significant "
            "evidence that medical charges differ between "
            "smokers and non-smokers."
        )

    else:

        st.success(
            "## Fail to Reject H₀"
        )

        st.write(
            f"The p-value ({p_value:.6f}) is greater than or equal "
            f"to α = {alpha}."
        )

        st.write(
            "**Conclusion:** There is insufficient statistical "
            "evidence to conclude that medical charges differ "
            "between smokers and non-smokers."
        )


    # ======================================
    # DECISION SUMMARY TABLE
    # ======================================

    st.markdown(
        "### 8. Statistical Decision Summary"
    )

    decision_summary = pd.DataFrame({
        "Component": [
            "Group 1",
            "Group 2",
            "Continuous Metric",
            "Group 1 Sample Size",
            "Group 2 Sample Size",
            "Group 1 Mean",
            "Group 2 Mean",
            "Shapiro-Wilk — Smokers",
            "Shapiro-Wilk — Non-Smokers",
            "Levene's Test",
            "Selected Test",
            "Test Statistic",
            "p-value",
            "Significance Level",
            "Final Decision"
        ],

        "Result": [
            "Smokers",
            "Non-Smokers",
            "Medical Charges",
            f"{len(smokers):,}",
            f"{len(non_smokers):,}",
            f"${smokers.mean():,.2f}",
            f"${non_smokers.mean():,.2f}",
            f"{shapiro_smokers.pvalue:.6f}",
            f"{shapiro_non_smokers.pvalue:.6f}",
            f"{levene_result.pvalue:.6f}",
            selected_test,
            f"{test_statistic:.4f}",
            f"{p_value:.6f}",
            "α = 0.05",
            (
                "Reject H₀"
                if p_value < alpha
                else "Fail to Reject H₀"
            )
        ]
    })

    st.dataframe(
        decision_summary,
        use_container_width=True,
        hide_index=True
    )


    st.divider()

        # ==========================================
    # HYPOTHESIS TEST 2: ONE-WAY ANOVA
    # ==========================================

    st.subheader(
        "Hypothesis Test 2: One-Way ANOVA"
    )

    st.caption(
        "Test whether average medical charges differ across "
        "the four geographic regions."
    )


    # ======================================
    # 1. VARIABLES
    # ======================================

    categorical_variable = "region"
    continuous_variable = "charges"

    st.write(
        "**Categorical Variable:** Geographic Region"
    )

    st.write(
        "**Continuous Variable:** Medical Charges"
    )

    st.write(
        "**Groups:** Northeast, Northwest, Southeast, Southwest"
    )


    # ======================================
    # 2. HYPOTHESES
    # ======================================

    st.markdown("### 1. Hypotheses")

    st.write(
        "**H₀ (Null Hypothesis):** "
        "The mean medical charges are equal across all four regions."
    )

    st.latex(
        r"H_0:\ \mu_{Northeast} = \mu_{Northwest} = "
        r"\mu_{Southeast} = \mu_{Southwest}"
    )

    st.write(
        "**H₁ (Alternative Hypothesis):** "
        "At least one region has a different mean medical charge."
    )

    st.info(
        "The alternative hypothesis does not specify which region "
        "is different. It only states that at least one regional "
        "mean differs from the others."
    )


    # ======================================
    # 3. GROUP SUMMARY
    # ======================================

    st.markdown("### 2. Group Summary")

    regions = [
        "northeast",
        "northwest",
        "southeast",
        "southwest"
    ]

    region_data = {}

    for region in regions:

        region_data[region] = df[
            df["region"] == region
        ]["charges"].dropna()


    # Display four regional summaries
    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.write("**Northeast**")

        st.metric(
            "Sample Size",
            f"{len(region_data['northeast']):,}"
        )

        st.write(
            f"Mean Charges: "
            f"**${region_data['northeast'].mean():,.2f}**"
        )

    with c2:

        st.write("**Northwest**")

        st.metric(
            "Sample Size",
            f"{len(region_data['northwest']):,}"
        )

        st.write(
            f"Mean Charges: "
            f"**${region_data['northwest'].mean():,.2f}**"
        )

    with c3:

        st.write("**Southeast**")

        st.metric(
            "Sample Size",
            f"{len(region_data['southeast']):,}"
        )

        st.write(
            f"Mean Charges: "
            f"**${region_data['southeast'].mean():,.2f}**"
        )

    with c4:

        st.write("**Southwest**")

        st.metric(
            "Sample Size",
            f"{len(region_data['southwest']):,}"
        )

        st.write(
            f"Mean Charges: "
            f"**${region_data['southwest'].mean():,.2f}**"
        )


    st.divider()


    # ======================================
    # 4. ONE-WAY ANOVA
    # ======================================

    st.markdown("### 3. One-Way ANOVA Test")

    st.write(
        "One-Way ANOVA compares the mean medical charges "
        "across the four independent geographic regions."
    )

    # Perform One-Way ANOVA
    anova_result = stats.f_oneway(
        region_data["northeast"],
        region_data["northwest"],
        region_data["southeast"],
        region_data["southwest"]
    )

    f_statistic = anova_result.statistic

    p_value_anova = anova_result.pvalue


    # ======================================
    # 5. ANOVA RESULTS
    # ======================================

    st.markdown("### 4. ANOVA Results")

    a1, a2 = st.columns(2)

    with a1:

        st.metric(
            "F-Statistic",
            f"{f_statistic:.4f}"
        )

    with a2:

        if p_value_anova < 0.000001:

            formatted_anova_p = "< 0.000001"

        else:

            formatted_anova_p = f"{p_value_anova:.6f}"

        st.metric(
            "p-value",
            formatted_anova_p
        )


    # ======================================
    # 6. SIGNIFICANCE LEVEL
    # ======================================

    alpha_anova = 0.05

    st.markdown(
        f"### 5. Final Conclusion at α = {alpha_anova}"
    )


    # ======================================
    # 7. FINAL DECISION
    # ======================================

    if p_value_anova < alpha_anova:

        st.error(
            "## Reject H₀"
        )

        st.write(
            f"The p-value ({formatted_anova_p}) is less than "
            f"α = {alpha_anova}."
        )

        st.write(
            "**Conclusion:** There is statistically significant "
            "evidence that average medical charges differ across "
            "the four geographic regions."
        )

        st.write(
            "The ANOVA result tells us that at least one regional "
            "mean is different, but ANOVA alone does not identify "
            "which specific regions differ."
        )

    else:

        st.success(
            "## Fail to Reject H₀"
        )

        st.write(
            f"The p-value ({formatted_anova_p}) is greater than "
            f"or equal to α = {alpha_anova}."
        )

        st.write(
            "**Conclusion:** There is insufficient statistical "
            "evidence to conclude that average medical charges "
            "differ across the four geographic regions."
        )


    # ======================================
    # 8. DECISION SUMMARY
    # ======================================

    st.markdown(
        "### 6. Statistical Decision Summary"
    )

    anova_summary = pd.DataFrame({
        "Component": [
            "Categorical Variable",
            "Continuous Variable",
            "Number of Groups",
            "Group 1",
            "Group 2",
            "Group 3",
            "Group 4",
            "F-Statistic",
            "p-value",
            "Significance Level",
            "Final Decision"
        ],

        "Result": [
            "Region",
            "Medical Charges",
            "4",
            "Northeast",
            "Northwest",
            "Southeast",
            "Southwest",
            f"{f_statistic:.4f}",
            formatted_anova_p,
            "α = 0.05",
            (
                "Reject H₀"
                if p_value_anova < alpha_anova
                else "Fail to Reject H₀"
            )
        ]
    })

    st.dataframe(
        anova_summary,
        use_container_width=True,
        hide_index=True
    )

# ==========================================
# TAB 3: STATISTICAL MODELING & DIAGNOSTICS
# ==========================================
with tab3:

    st.header("📈 Statistical Modeling & Diagnostic Checks")

    # ======================================
    # 1. MODEL FORMULATION
    # ======================================

    st.subheader("1. Multiple Linear Regression Model")

    st.write(
        "A multiple linear regression model is used to estimate "
        "medical charges using age, BMI, number of children, "
        "smoking status, sex, and geographic region."
    )

    st.write("**Model:**")

    st.latex(
        r"Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + "
        r"\cdots + \beta_kX_k + \epsilon"
    )

    st.write(
        "**For this dataset:**"
    )

    st.code(
        "charges ~ age + bmi + children + smoker + sex + region + bmi × smoker",
        language="text"
    )


    # ======================================
    # PREPARE DATA FOR OLS
    # ======================================

    model_data = df[
        [
            "charges",
            "age",
            "bmi",
            "children",
            "smoker",
            "sex",
            "region"
        ]
    ].dropna().copy()


    # Convert categorical variables into dummy variables
    model_data["smoker_yes"] = (
        model_data["smoker"] == "yes"
    ).astype(int)

    model_data["sex_male"] = (
        model_data["sex"] == "male"
    ).astype(int)

    region_dummies = pd.get_dummies(
        model_data["region"],
        prefix="region",
        drop_first=True,
        dtype=int
    )

    model_data = pd.concat(
        [
            model_data,
            region_dummies
        ],
        axis=1
    )


    # ======================================
    # INTERACTION TERM
    # ======================================

    model_data["bmi_smoker"] = (
        model_data["bmi"]
        * model_data["smoker_yes"]
    )


    # ======================================
    # DEFINE X AND Y
    # ======================================

    predictor_columns = [
        "age",
        "bmi",
        "children",
        "smoker_yes",
        "sex_male"
    ] + region_dummies.columns.tolist() + [
        "bmi_smoker"
    ]

    X = model_data[predictor_columns].astype(float)

    y = model_data["charges"].astype(float)


    # Add intercept β0
    X = sm.add_constant(X)


    # ======================================
    # FIT OLS MODEL
    # ======================================

    ols_model = sm.OLS(
        y,
        X
    ).fit()


    # ======================================
    # 2. OVERALL MODEL FIT
    # ======================================

    st.subheader("2. Overall Model Fit")

    fit_col1, fit_col2 = st.columns(2)

    with fit_col1:

        st.metric(
            "R-squared",
            f"{ols_model.rsquared:.4f}"
        )

    with fit_col2:

        st.metric(
            "Adjusted R-squared",
            f"{ols_model.rsquared_adj:.4f}"
        )


    st.write(
        "**R²** measures the proportion of variation in medical "
        "charges explained by the predictors in the model."
    )

    st.write(
        "**Adjusted R²** adjusts R² for the number of predictors "
        "included in the model and is useful when multiple predictors "
        "are used."
    )


    # ======================================
    # 3. PARAMETER INTERPRETATION
    # ======================================

    st.subheader("3. Parameter Estimates & Interpretation")

    st.write(
        "The table below reports the estimated regression "
        "coefficients, standard errors, p-values, and 95% "
        "confidence intervals."
    )


    coefficient_table = pd.DataFrame({
        "Coefficient (β)": ols_model.params,
        "Std. Error": ols_model.bse,
        "p-value": ols_model.pvalues,
        "95% CI Lower": ols_model.conf_int()[0],
        "95% CI Upper": ols_model.conf_int()[1]
    })

    coefficient_table.index.name = "Predictor"

    st.dataframe(
        coefficient_table.style.format({
            "Coefficient (β)": "{:,.4f}",
            "Std. Error": "{:,.4f}",
            "p-value": "{:.6f}",
            "95% CI Lower": "{:,.4f}",
            "95% CI Upper": "{:,.4f}"
        }),
        use_container_width=True
    )


    st.write(
        "**Coefficient (β):** Estimated change in medical charges "
        "associated with a one-unit increase in the predictor, "
        "holding the other predictors constant."
    )

    st.write(
        "**p-value:** Tests whether the corresponding coefficient "
        "is statistically significantly different from zero."
    )

    st.write(
        "**95% Confidence Interval:** Gives the estimated range "
        "within which the population coefficient is expected to lie "
        "with 95% confidence."
    )


    # ======================================
    # SIGNIFICANCE INTERPRETATION
    # ======================================

    st.markdown("### Coefficient Significance")

    significant_predictors = []

    for predictor in ols_model.params.index:

        if predictor == "const":
            continue

        if ols_model.pvalues[predictor] < 0.05:

            significant_predictors.append(predictor)

    if significant_predictors:

        st.success(
            "Statistically significant predictors at α = 0.05: "
            + ", ".join(significant_predictors)
        )

    else:

        st.info(
            "No predictors are statistically significant at α = 0.05."
        )


    st.divider()


    # ======================================
    # 4. GAUSS-MARKOV DIAGNOSTIC CHECKS
    # ======================================

    st.subheader(
        "4. Gauss-Markov Diagnostic Checks"
    )

    st.write(
        "The following diagnostics are used to assess important "
        "regression assumptions."
    )


    # ======================================
    # 4.1 LINEARITY & HOMOSCEDASTICITY
    # ======================================

    st.markdown(
        "### 4.1. Linearity & Homoscedasticity"
    )

    st.write(
        "Residuals are plotted against fitted values. "
        "A desirable plot shows residuals randomly scattered "
        "around zero without a systematic pattern."
    )

    fitted_values = ols_model.fittedvalues
    residuals = ols_model.resid

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    sns.scatterplot(
        x=fitted_values,
        y=residuals,
        alpha=0.6,
        color="steelblue",
        edgecolor="black",
        s=45,
        ax=ax
    )

    ax.axhline(
        0,
        color="red",
        linestyle="--",
        linewidth=2
    )

    ax.set_title(
        "Residuals vs Fitted Values",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Fitted Values",
        fontsize=11
    )

    ax.set_ylabel(
        "Residuals",
        fontsize=11
    )

    st.pyplot(
        fig,
        use_container_width=False
    )

    st.info(
        "Interpretation: Randomly scattered residuals around zero "
        "support linearity and approximately constant variance. "
        "A funnel shape or systematic pattern may indicate "
        "heteroscedasticity or non-linearity."
    )


    # ======================================
    # 4.2 NORMALITY OF RESIDUALS
    # ======================================

    st.markdown(
        "### 4.2. Normality of Residuals — Q-Q Plot"
    )

    st.write(
        "The Q-Q plot compares the distribution of the regression "
        "residuals with a theoretical normal distribution."
    )

    fig, ax = plt.subplots(
        figsize=(6, 4)
    )

    sm.qqplot(
        residuals,
        line="45",
        fit=True,
        ax=ax
    )

    ax.set_title(
        "Normal Q-Q Plot of Regression Residuals",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Theoretical Quantiles"
    )

    ax.set_ylabel(
        "Sample Quantiles"
    )

    st.pyplot(
        fig,
        use_container_width=False
    )


    # ======================================
    # JARQUE-BERA TEST
    # ======================================

    jb_statistic, jb_pvalue, skewness, kurtosis = jarque_bera(
        residuals
    )

    st.markdown(
        "### Jarque-Bera Normality Test"
    )

    jb1, jb2 = st.columns(2)

    with jb1:

        st.metric(
            "Jarque-Bera Statistic",
            f"{jb_statistic:.4f}"
        )

    with jb2:

        if jb_pvalue < 0.000001:

            formatted_jb_p = "< 0.000001"

        else:

            formatted_jb_p = f"{jb_pvalue:.6f}"

        st.metric(
            "p-value",
            formatted_jb_p
        )


    st.write(
        "**H₀:** Regression residuals are normally distributed."
    )

    st.write(
        "**H₁:** Regression residuals are not normally distributed."
    )


    if jb_pvalue < 0.05:

        st.error(
            "p < 0.05 → Reject H₀. "
            "There is statistical evidence that the residuals "
            "are not normally distributed."
        )

    else:

        st.success(
            "p ≥ 0.05 → Fail to Reject H₀. "
            "There is no significant evidence of non-normality "
            "in the residuals."
        )


    st.write(
        f"**Residual Skewness:** {skewness:.4f}"
    )

    st.write(
        f"**Residual Kurtosis:** {kurtosis:.4f}"
    )


    st.divider()


    # ======================================
    # 4.3 MULTICOLLINEARITY — VIF
    # ======================================

    st.markdown(
        "### 4.3. Multicollinearity — Variance Inflation Factor (VIF)"
    )

    st.write(
        "VIF measures how strongly each continuous predictor "
        "is linearly related to the other continuous predictors."
    )

    st.write(
        "For this requirement, VIF is calculated for the "
        "continuous predictors: age, BMI, and children."
    )


    continuous_predictors = [
        "age",
        "bmi",
        "children"
    ]

    vif_data = model_data[
        continuous_predictors
    ].astype(float)

    vif_table = pd.DataFrame()

    vif_table["Predictor"] = continuous_predictors

    vif_table["VIF"] = [
        variance_inflation_factor(
            vif_data.values,
            i
        )
        for i in range(
            vif_data.shape[1]
        )
    ]


    st.dataframe(
        vif_table.style.format({
            "VIF": "{:.4f}"
        }),
        use_container_width=False,
        hide_index=True
    )


    st.write(
        "**VIF ≈ 1:** Very little multicollinearity."
    )

    st.write(
        "**VIF between 1 and 5:** Usually considered acceptable."
    )

    st.write(
        "**VIF > 5:** Potential multicollinearity concern."
    )

    st.write(
        "**VIF > 10:** Strong multicollinearity concern."
    )


    st.divider()


    # ======================================
    # 5. LIVE INTERACTIVE PREDICTION
    # ======================================

    st.subheader(
        "5. Live Interactive Prediction"
    )

    st.write(
        "Enter patient characteristics to generate a predicted "
        "medical charge from the fitted OLS model."
    )


    left_col, right_col = st.columns([1.2, 1])


    # ======================================
    # INPUT CONTROLS
    # ======================================

    with right_col:

        st.markdown(
            "### Patient Inputs"
        )

        input_age = st.slider(
            "Age",
            18,
            100,
            35,
            key="prediction_age"
        )

        input_bmi = st.slider(
            "BMI",
            10.0,
            50.0,
            27.5,
            key="prediction_bmi"
        )

        input_children = st.selectbox(
            "Children",
            [0, 1, 2, 3, 4, 5],
            index=0,
            key="prediction_children"
        )

        input_smoker = st.selectbox(
            "Smoker",
            ["yes", "no"],
            index=1,
            key="prediction_smoker"
        )

        input_sex = st.selectbox(
            "Sex",
            ["male", "female"],
            index=0,
            key="prediction_sex"
        )

        input_region = st.selectbox(
            "Region",
            [
                "northeast",
                "northwest",
                "southeast",
                "southwest"
            ],
            index=0,
            key="prediction_region"
        )


    # ======================================
    # CREATE NEW OBSERVATION
    # ======================================

    new_patient = pd.DataFrame({
        "age": [input_age],
        "bmi": [input_bmi],
        "children": [input_children],
        "smoker_yes": [
            1 if input_smoker == "yes" else 0
        ],
        "sex_male": [
            1 if input_sex == "male" else 0
        ]
    })


    # Add region dummy variables
    for region_column in region_dummies.columns:

        region_name = region_column.replace(
            "region_",
            ""
        )

        new_patient[region_column] = (
            1
            if input_region == region_name
            else 0
        )


    # Interaction term
    new_patient["bmi_smoker"] = (
        input_bmi
        * (
            1
            if input_smoker == "yes"
            else 0
        )
    )


    # Make sure columns are in exactly the same order
    new_patient = new_patient[
        predictor_columns
    ].astype(float)

    new_patient = sm.add_constant(
        new_patient,
        has_constant="add"
    )

    new_patient = new_patient[
        X.columns
    ]


    # ======================================
    # PREDICTION
    # ======================================

    prediction_result = ols_model.get_prediction(
        new_patient
    )

    prediction_summary = prediction_result.summary_frame(
        alpha=0.05
    )

    predicted_charges = prediction_summary[
        "mean"
    ].iloc[0]

    confidence_lower = prediction_summary[
        "mean_ci_lower"
    ].iloc[0]

    confidence_upper = prediction_summary[
        "mean_ci_upper"
    ].iloc[0]

    prediction_lower = prediction_summary[
        "obs_ci_lower"
    ].iloc[0]

    prediction_upper = prediction_summary[
        "obs_ci_upper"
    ].iloc[0]


    # ======================================
    # DISPLAY PREDICTION
    # ======================================

    with left_col:

        st.markdown(
            "### Prediction Results"
        )

        st.metric(
            "Predicted Medical Charges",
            f"${predicted_charges:,.2f}"
        )

        st.write(
            f"**95% Confidence Interval:** "
            f"[${confidence_lower:,.2f}, "
            f"${confidence_upper:,.2f}]"
        )

        st.write(
            f"**95% Prediction Interval:** "
            f"[${prediction_lower:,.2f}, "
            f"${prediction_upper:,.2f}]"
        )

        st.info(
            "The confidence interval estimates the uncertainty "
            "around the mean predicted charge. The prediction "
            "interval is wider because it also accounts for "
            "individual-level variability."
        )


    # ======================================
    # MODEL SUMMARY
    # ======================================

    with st.expander(
        "View Complete OLS Model Summary"
    ):

        st.text(
            ols_model.summary().as_text()
        )