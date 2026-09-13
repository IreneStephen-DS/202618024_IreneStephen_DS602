# M.Sc. Data Science — Semester 1

## Lab-4: Applied Statistical Modeling & Interactive Web Dashboard

### Course

**Statistical Modeling with Python**

### Dataset

**Medical Insurance Costs Dataset**

### Tools & Technologies

**Python, Pandas, NumPy, SciPy, Statsmodels, Scikit-learn, Matplotlib, Seaborn, Plotly, Streamlit, VS Code**

---

## Project Overview

This project performs an end-to-end statistical analysis of medical insurance costs, including exploratory data analysis, hypothesis testing, multiple linear regression using OLS, regression diagnostics, multicollinearity analysis, and an interactive Streamlit dashboard for data exploration, statistical testing, prediction, and diagnostics.

---

## Assignment Objectives

* **Data Understanding:** Examined the structure, variables, data types, missing values, duplicate records, and basic characteristics of the insurance dataset.
* **Descriptive Statistics:** Calculated mean, median, standard deviation, interquartile range (IQR), skewness, and kurtosis for numerical variables.
* **Distribution Analysis:** Visualized numerical variables using histograms and KDE plots to understand their distributions and identify skewness or unusual patterns.
* **Bivariate Analysis:** Investigated relationships between numerical variables using scatter plots and analyzed patterns across categorical groups.
* **Correlation Analysis:** Constructed a correlation matrix to measure the strength and direction of relationships among numerical variables.
* **Hypothesis Test 1:** Compared medical charges between smokers and non-smokers using an appropriate two-group statistical test.
* **Normality Testing:** Applied the Shapiro-Wilk test to assess whether the groups followed an approximately normal distribution before selecting the appropriate comparison test.
* **Equal Variance Testing:** Applied Levene's test to determine whether the variances of the two comparison groups were statistically similar.
* **Two-Sample Testing:** Applied an independent two-sample t-test when normality assumptions were satisfied and used the Mann-Whitney U test when the data were non-normal.
* **Hypothesis Decision:** Evaluated p-values at a significance level of α = 0.05 and reported whether the null hypothesis was rejected or failed to be rejected.
* **Hypothesis Test 2:** Performed a Chi-Square test of independence to investigate the association between smoking status and geographic region.
* **Chi-Square Analysis:** Generated a contingency table and evaluated the Chi-Square statistic, degrees of freedom, expected frequencies, p-value, and statistical conclusion.
* **Regression Modeling:** Developed a multiple linear regression model using Ordinary Least Squares (OLS) with medical charges as the dependent variable and relevant numerical and categorical predictors as independent variables.
* **Categorical Encoding:** Represented categorical predictors appropriately for inclusion in the OLS regression model.
* **Coefficient Interpretation:** Interpreted estimated regression coefficients in terms of their direction, magnitude, and relationship with the dependent variable while controlling for other predictors.
* **Statistical Significance:** Evaluated regression coefficient p-values to determine which predictors provide statistically significant evidence of an association with medical charges.
* **Confidence Intervals:** Reported and interpreted 95% confidence intervals for the estimated regression coefficients.
* **Model Fit:** Evaluated overall model performance using R² and Adjusted R².
* **Overall Regression Significance:** Evaluated the overall statistical significance of the regression model using the appropriate model-level statistical test.
* **Residual Analysis:** Calculated and analyzed regression residuals, defined as the difference between observed and fitted values.
* **Linearity Diagnostic:** Used a residuals-versus-fitted-values plot to assess whether the linearity assumption was reasonably satisfied.
* **Homoscedasticity Diagnostic:** Examined the residual spread across fitted values to assess whether the constant-variance assumption was reasonably satisfied.
* **Residual Normality:** Generated a Q-Q plot to visually assess whether the regression residuals were approximately normally distributed.
* **Jarque-Bera/Omnibus Test:** Applied a formal residual normality test and reported its test statistic and p-value.
* **Multicollinearity Analysis:** Calculated Variance Inflation Factor (VIF) values for continuous predictors to identify potential multicollinearity problems.
* **Model Diagnostics:** Combined residual plots, normality testing, and VIF analysis to evaluate the reliability and assumptions of the fitted regression model.
* **Interactive Dashboard:** Developed an interactive Streamlit web application to convert the statistical analysis into a user-friendly data science dashboard.
* **Data Exploration Tab:** Implemented interactive filters, dataset summary statistics, distribution visualizations, scatter plots, and correlation analysis for exploratory data analysis.
* **Hypothesis Testing Lab Tab:** Implemented dynamic variable selection for categorical and numerical variables with automatic statistical test execution, test statistics, p-values, and hypothesis conclusions.
* **Live Prediction & Diagnostics Tab:** Implemented interactive user inputs for generating real-time medical charge predictions using the fitted OLS model.
* **Prediction Interval:** Provided a 95% prediction/confidence interval around the model prediction to represent uncertainty associated with predicted medical charges.
* **Interactive Diagnostics:** Displayed regression residual diagnostic plots and statistical diagnostic results directly within the Streamlit dashboard.
* **User-Friendly Interface:** Designed the dashboard with interactive controls, organized tabs, clear statistical outputs, and visualizations for accessible interpretation of the analysis.
* **Application Testing:** Tested the Streamlit application to ensure that filtering, statistical calculations, visualizations, prediction, and diagnostic outputs function correctly.

---

## Dataset Variables

| Variable   | Type        | Description                      |
| ---------- | ----------- | -------------------------------- |
| `age`      | Numerical   | Age of the insurance beneficiary |
| `sex`      | Categorical | Sex of the beneficiary           |
| `bmi`      | Numerical   | Body Mass Index                  |
| `children` | Numerical   | Number of children/dependents    |
| `smoker`   | Categorical | Smoking status                   |
| `region`   | Categorical | Residential region               |
| `charges`  | Numerical   | Medical insurance charges        |

---

## Hypothesis Testing

### Hypothesis Test 1 — Two-Group Comparison

**Research Question:** Do medical charges differ significantly between smokers and non-smokers?

**H0:** There is no statistically significant difference in medical charges between smokers and non-smokers.

**H1:** There is a statistically significant difference in medical charges between smokers and non-smokers.

**Significance Level:** α = 0.05

**Procedure:** Shapiro-Wilk normality test → Levene's equal-variance test → Independent two-sample t-test or Mann-Whitney U test → Statistical conclusion.

### Hypothesis Test 2 — Chi-Square Test

**Research Question:** Is smoking status associated with geographic region?

**H0:** Smoking status and region are statistically independent.

**H1:** Smoking status and region are statistically associated.

**Significance Level:** α = 0.05

**Procedure:** Contingency table → Chi-Square test → Test statistic → Degrees of freedom → p-value → Statistical conclusion.

---

## Regression Model

The multiple linear regression model is formulated as:

$$
Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \cdots + \beta_kX_k + \epsilon
$$

where **Y represents medical charges**, **X represents the selected numerical and categorical predictors**, **β represents the estimated regression coefficients**, and **ε represents the random error term**.

---

## Regression Evaluation

The OLS model is evaluated using **regression coefficients, standard errors, t-statistics, p-values, 95% confidence intervals, R², Adjusted R², and overall model significance**.

---

## Regression Diagnostics

The fitted model is diagnosed using **Residuals vs Fitted plots for linearity and homoscedasticity, Q-Q plots and Jarque-Bera/Omnibus testing for residual normality, and VIF for multicollinearity**.

---

## Streamlit Dashboard

The interactive dashboard consists of three main tabs:

### Tab 1 — Data Exploration

Provides **interactive sidebar filters, dataset summaries, descriptive statistics, histograms/KDE plots, scatter plots, and correlation visualizations**.

### Tab 2 — Hypothesis Testing Lab

Provides **dynamic categorical and numerical variable selection, automatic hypothesis-test selection, test statistics, p-values, α = 0.05 decisions, and clear statistical conclusions**.

### Tab 3 — Live Prediction & Diagnostics

Provides **interactive predictor inputs, real-time OLS medical-charge predictions, 95% prediction/confidence intervals, residual diagnostic plots, normality results, and VIF information**.

---

## Project Workflow

**Data Collection → Data Understanding → Data Validation → Descriptive Statistics → EDA → Hypothesis Testing → OLS Regression → Coefficient Interpretation → Residual Diagnostics → VIF Analysis → Streamlit Dashboard → Live Prediction → Final Interpretation**

---

## Project Structure

```text
Lab_4_Statistical_Modeling/
│
├── insurance.csv
├── analysis.py
├── app.py
├── requirements.txt
├── README.md
│
└── screenshots/
    ├── data_exploration.png
    ├── hypothesis_testing.png
    └── prediction_diagnostics.png
```

---

## Installation

Create and activate a Python virtual environment, then install the required packages using:

```text
pip install -r requirements.txt
```

---

## Running the Statistical Analysis

Run the analysis script from the VS Code terminal using:

```text
python analysis.py
```

---

## Running the Streamlit Dashboard

Launch the interactive web application using:

```text
streamlit run app.py
```

The application can then be accessed through the local Streamlit URL displayed in the terminal.

---

## Expected Outcome

The final project provides a complete statistical modeling workflow that explains the characteristics of medical insurance data, statistically evaluates relationships between variables, develops and validates an OLS regression model, and delivers the resulting analysis through an interactive Streamlit web dashboard with live prediction capabilities.

---

## Learning Outcomes Achieved

This project demonstrates the ability to **perform exploratory data analysis, formulate and evaluate parametric and non-parametric hypothesis tests, conduct Chi-Square analysis, build and interpret OLS regression models, evaluate Gauss-Markov assumptions, diagnose multicollinearity, interpret statistical inference, and deploy an interactive statistical modeling application using Streamlit**.

---

## Author

**Name:** __________________________

**Roll No.:** _______________________

**Program:** M.Sc. Data Science

**Semester:** I

**Course:** Statistical Modeling with Python

**Lab:** Lab-4 — Applied Statistical Modeling & Interactive Web Dashboard

**Academic Year:** __________________
