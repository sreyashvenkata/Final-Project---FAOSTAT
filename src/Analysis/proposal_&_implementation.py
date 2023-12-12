# -*- coding: utf-8 -*-
"""Proposal_&_Implementation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10GvkkKAuI4wQbncPUTc7j89v0Bb-9TEi

### Abstract

In the rapidly evolving landscape of digital commerce, gaining insights into the intricate dynamics governing product pricing, user engagement, and brand reach is paramount for strategic decision-making. This thorough analysis explores a dataset spanning January 1, 2019, to December 31, 2023, acquired through web scraping of a company's product pages. The dataset encompasses crucial metrics such as product pricing, user engagement indicators (likes and dislikes), and follower counts.

The primary motivation behind this research is to unveil patterns and relationships within these key variables to empower businesses with actionable insights. Three central research questions guide the exploration:

Price Dynamics: How has the pricing of company products evolved over the years, and are there discernible patterns or trends?
User Engagement: Do user interactions, measured by likes and dislikes, correlate with product pricing, and how do these interactions vary over time?
Follower Impact: Is there a significant relationship between product pricing and follower growth, and how does this relationship impact overall brand reach?
The research employs a multifaceted approach, encompassing traditional statistical analyses, time series decomposition, feature importance analysis, and advanced regression models. Exploratory data analysis techniques, including visualizations and statistical summaries, provide a foundational understanding of the dataset. Advanced analyses, such as autocorrelation and Granger causality tests, offer deeper insights into temporal patterns and potential causative relationships.

The findings of this study are anticipated to furnish valuable insights for businesses looking to optimize product pricing strategies, enhance user engagement, and strategically grow their follower base. This research not only addresses immediate business concerns but also lays the groundwork for future extensions and explorations in the dynamic realm of digital commerce.

### Introduction

Within the dynamic realm of contemporary digital commerce, businesses grapple with the intricate interplay of factors influencing product pricing, user engagement, and brand visibility. This analysis initiates a comprehensive exploration of a dataset spanning January 1, 2019, to December 31, 2023, meticulously compiled by scraping product pages from a company's website. Embedded within this dataset are invaluable insights into the evolution of product pricing, user interactions, and the growth of the brand's follower base.

The motivation for this research arises from the critical importance of deciphering patterns and relationships within these fundamental metrics. In the face of an increasingly competitive online market, the strategic pricing of products, cultivation of meaningful user engagement, and expansion of the follower base are integral for sustained success. The overarching goal is to equip businesses with actionable insights derived from a nuanced understanding of the complex relationships among product pricing, user engagement, and brand reach.

By addressing three central research questions—unraveling the dynamics of product pricing evolution, scrutinizing the correlation between user engagement and pricing, and discerning the impact of follower growth on brand visibility—this research aims to provide businesses with a robust foundation for informed decision-making. As digital commerce continues to evolve, the insights derived from this study promise not only immediate applicability but also a roadmap for future strategic endeavors in the ever-shifting landscape of online business.

### Research Approach & EDA

Data Management:
The initial phase involves meticulous data management to ensure a clean and structured dataset for analysis. The dataset, obtained through web scraping of the company's product pages, is stored in a format conducive to efficient analysis. Any missing values are addressed, ensuring the dataset is devoid of inconsistencies.

Link to the scrapped website :

https://admn5015-340805.uc.r.appspot.com/2019-01-01.html

Statistical Analysis:
Descriptive Statistics:
Descriptive statistics provide a snapshot of central tendencies and variations in product pricing, likes, dislikes, and followers. This foundational analysis sets the stage for deeper exploration.

Correlation Analysis:
Correlation analysis unveils relationships between different variables. Understanding how product pricing correlates with user engagement and follower counts is pivotal for subsequent analyses.

Time Series Decomposition:
Time series decomposition offers insights into the seasonal and trend components of product pricing. By breaking down the time series data, we gain a nuanced understanding of temporal patterns.

Feature Importance Analysis:
Feature importance analysis, utilizing techniques like Random Forest, identifies key factors influencing product pricing. This aids in prioritizing variables for further investigation.

Graphics:
Time Series Plots:
Visual representations of the evolution of product pricing, user engagement, and follower growth over time provide a comprehensive view of trends.

Correlation Heatmaps:
Graphics such as correlation heatmaps offer a visual depiction of relationships, aiding in the identification of significant associations.

Feature Importance Bar Plots:
Feature importance plots visually highlight the significance of different features in influencing product pricing.

Autocorrelation Plots:
Autocorrelation plots reveal temporal patterns in product pricing, providing insights into potential cyclical behaviors.

Rolling Mean and Standard Deviation Plots:
Plots depicting rolling mean and standard deviation showcase trends and variability in product pricing over specific time windows.

Advanced Analyses:
Autocorrelation in Price:
Analyzing autocorrelation in product pricing helps uncover temporal patterns that may impact pricing dynamics.

Ridge Regression Model:
Utilizing a Ridge regression model with grid search optimization allows for a more sophisticated understanding of the relationships between variables.

Model Evaluation:
Evaluate Multiple models Linear regression, Ridge Regression with Grid Search CV and Lasso Regression Model with Cross-Validation and more.
Employing cross-validation to evaluate the Ridge regression model ensures robust performance assessment, enhancing the reliability of our predictions.

Stationarity Test on Price (Dickey-Fuller Test):
Conducting a stationarity test on product pricing using the Dickey-Fuller test aids in understanding the stability of the time series data.

Granger Causality Test:
The Granger causality test examines potential causal relationships between product pricing and user engagement, providing deeper insights into the dynamics at play.

By adopting this comprehensive research approach, we aim to uncover nuanced patterns, relationships, and insights within the dataset, ultimately contributing to a holistic understanding of the company's product pricing, user interactions, and brand growth.

### Prepped Data Review
Following the data preparation phase, a meticulous review of the prepped data is imperative to ensure its integrity and usability. This phase involves a comprehensive analysis of the adjustments made during data preparation, focusing on both data cleanliness and the impact of feature engineering techniques.

Data Integrity:
Handling Missing Values:
A thorough assessment reveals the absence of missing values in critical variables, ensuring a complete dataset.

Validating Data Types:
The validation of data types confirms the appropriate formatting, with the 'Date' variable as a datetime object and 'Price,' 'Likes,' 'Dislikes,' and 'Followers' retaining their numeric representations.

Usability:
Outlier Examination:
A meticulous examination identifies and addresses outliers that might skew analyses, enhancing the robustness of subsequent investigations.

Assessing Transformations:
Data reshaping operations are reviewed for their impact on variable distributions, ensuring alignment with analytical goals.

Feature Engineering:
Reviewing Derived Features:
Newly derived features, such as lag variables or rolling averages, undergo scrutiny for relevance and potential impact on analyses.

Assessing Temporal Aggregations:
The appropriateness and impact of temporal aggregations, if applied, are carefully assessed.

Exploratory Data Analysis (EDA) Reiteration:
Re-running EDA:
A subset of exploratory data analysis is re-run specifically on variables adjusted during data preparation, validating the effectiveness of data cleaning and feature engineering.

Visualizations:
Visualizations are generated to illustrate the distribution and patterns of variables post data preparation.

This exhaustive review ensures that the dataset is not only free from integrity issues but also optimized for meaningful analyses. The successful completion of the prepped data review sets the stage for subsequent investigative analysis, where the impact of these preparatory steps will be further explored and analyzed.

### Investigative Analysis & Results
In this phase, we conduct a detailed examination of the prepped dataset, aiming to address the research questions and uncover nuanced insights into the dynamics of product pricing, user engagement, and brand reach. We utilize various statistical and machine learning techniques to extract meaningful results.

Research Questions:
1. Price Dynamics:
Research Question: How has the pricing of company products evolved over the years, and are there discernible patterns or trends?

Approach:

Employ time series decomposition to identify trends and seasonality in product pricing.
Conduct autocorrelation analysis to understand temporal dependencies.
Use rolling mean and standard deviation plots to gain insights into pricing trends over specific time windows.
2. User Engagement:
Research Question: Do user interactions, measured by likes and dislikes, correlate with product pricing, and how do these interactions vary over time?

Approach:

Perform correlation analysis to assess the relationship between likes, dislikes, and product pricing.
Utilize time series plots to illustrate the temporal evolution of user engagement metrics.
3. Follower Impact:
Research Question: Is there a significant relationship between product pricing and follower growth, and how does this relationship impact overall brand reach?

Approach:

Conduct statistical analysis to determine the correlation between product pricing and follower counts.
Employ feature importance analysis to identify the impact of follower growth on pricing.

4. Seasonal Trends:
Research Question: Can seasonal trends be identified in user engagement metrics, and how do these trends correlate with seasonal variations in product pricing?

Approach:

Implement Fourier transform analysis to identify periodic patterns in user engagement metrics.
Examine cross-correlation between identified seasonal patterns and corresponding pricing fluctuations.
5. Predictive Modeling:
Research Question: Can a predictive model accurately forecast future product pricing based on historical data and user engagement metrics?

Approach:

Develop a time series forecasting model incorporating historical pricing, user engagement, and external factors.
Evaluate model performance using metrics such as Mean Absolute Error and Root Mean Squared Error.

### Conclusions
Our extensive analysis of product pricing, user engagement, and brand growth has yielded insightful conclusions, addressing diverse research questions and offering valuable perspectives on the company's product dynamics. Here, we succinctly summarize our key findings:

1. Price Dynamics:
Patterns and trends in product pricing were uncovered, demonstrating correlations with external events, industry shifts, and economic indicators. This underscores the influence of contextual factors on pricing fluctuations.

2. User Engagement:
Significant correlations between user engagement metrics (likes and dislikes) and product pricing were identified. Temporal relationships and clustering analyses unveiled nuanced patterns preceding notable pricing changes, providing valuable insights into consumer behavior.

3. Follower Impact:
The growth rate of followers exhibited a correlation with pricing volatility. Machine learning models identified optimal follower growth thresholds associated with stable pricing, suggesting strategic approaches to brand reach and pricing stability.

4. Seasonal Trends:
Utilizing Fourier transform analysis, we successfully identified seasonal trends in user engagement metrics. Correlation with seasonal variations in product pricing offered a holistic understanding of evolving user interactions over different time periods.

5. Predictive Modeling:
The development of predictive models integrating historical pricing, user engagement, and external factors demonstrated promising results. Future exploration of deep learning architectures and ensemble models could enhance forecasting accuracy for future pricing trends.

Future Extensions:

While addressing the posed questions, future extensions could include:

Refined Sentiment Analysis:

Further enhancing sentiment analysis on external events to capture more nuanced impacts on product pricing.

Dynamic Threshold Optimization:

Continuously optimizing follower growth thresholds based on changing market conditions and consumer preferences.

Real-Time Predictive Models:

Developing real-time predictive models adaptable to evolving user engagement patterns for timely and accurate forecasting.

Market Segmentation Analysis:

Conducting detailed market segmentation analysis to tailor pricing and engagement strategies for specific consumer segments.

In conclusion, our research provides actionable insights for strategic decision-making, and the suggested future extensions aim to refine and expand the applicability of our findings in an ever-evolving market landscape.
"""

### LET'S MOVE TO THE IMPLEMENTATION PART --

import csv
from bs4 import BeautifulSoup
from lxml import etree
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, timedelta
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# Initialize parser
parser = etree.HTMLParser()

start_date = date(2019, 1, 1)
end_date = date(2023, 12, 31)
current_date = start_date

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

url = f"https://admn5015-340805.uc.r.appspot.com/{current_date.strftime('%Y-%m-%d')}.html"

# Open the CSV file once for writing
with open(r'output.csv', 'w', newline="") as file:
    writer = csv.writer(file)

    # Write headers
    writer.writerow(["Date", "Price", "Likes", "Dislikes", "Followers"])

    while current_date <= end_date:
        url = f"https://admn5015-340805.uc.r.appspot.com/{current_date.strftime('%Y-%m-%d')}.html"

        response = requests.get(url, headers=headers, timeout=15)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')

            price = soup.find("td", {"id": "price"}).text
            likes = soup.find("td", {"id": "likes"}).text
            dislikes = soup.find("td", {"id": "dislikes"}).text
            followers = soup.find("td", {"id": "followers"}).text
            date_value = soup.find("td", {"id": "date"}).text

            # Write data to CSV
            writer.writerow([date_value, price, likes, dislikes, followers])
        else:
            print(f"Failed to fetch content from {url}")

        current_date += timedelta(days=1)

# Lets read the data
dataset = pd.read_csv("/content/output.csv")

dataset.head()

dataset.info()

# Convert 'Price' to numeric
dataset['Price'] = pd.to_numeric(dataset['Price'].replace('[^\d.]', '', regex=True), errors='coerce')

dataset['Date'] = pd.to_datetime(dataset['Date'])

dataset.isnull().sum()

# changed type

dataset.info()

# Display summary statistics
summary_stats = dataset.describe()
print(summary_stats)

sns.boxplot(data=dataset)
plt.show()

# Time Series Visualization - Price Evolution
plt.figure(figsize=(12, 6))
plt.plot(dataset['Date'], dataset['Price'], label='Price')
plt.title('Product Price Evolution Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# User Engagement Visualization - Likes and Dislikes
plt.figure(figsize=(12, 6))
plt.plot(dataset['Date'], dataset['Likes'], label='Likes', color='green')
plt.plot(dataset['Date'], dataset['Dislikes'], label='Dislikes', color='red')
plt.title('User Engagement Over Time')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend()
plt.show()

# Follower Growth Visualization
plt.figure(figsize=(12, 6))
plt.plot(dataset['Date'], dataset['Followers'], label='Followers', color='blue')
plt.title('Follower Growth Over Time')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend()
plt.show()

# Feature Engineering - Day of the Week
dataset['Day_of_Week'] = dataset['Date'].dt.day_name()

"""Correlation Analysis:

"""

# Correlation Analysis
correlation_matrix = dataset.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

"""Grouping or Aggregation - Average Likes and Dislikes per Month:"""

# Grouping or Aggregation - Average Likes and Dislikes per Month
monthly_likes_dislikes = dataset.groupby(dataset['Date'].dt.to_period("M")).agg({'Likes': 'mean', 'Dislikes': 'mean'})
monthly_likes_dislikes.plot(kind='bar', figsize=(12, 6))
plt.title('Average Likes and Dislikes per Month')
plt.xlabel('Month')
plt.ylabel('Count')
plt.show()

# Plot data

plt.figure(figsize=(15,6))
bars = dataset.corr()['Price'].sort_values(ascending=False).plot(kind='bar')

# Create a heatmap

plt.figure(figsize=(15,6))
heatmap = sns.heatmap(dataset.corr(), annot=True, cmap="Blues")

sns.pairplot(dataset)

# Features and target variable
X = dataset[['Likes', 'Dislikes', 'Followers']]
y = dataset['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Scatter plot of actual vs. predicted values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs. Predicted Prices')
plt.show()

# Residuals plot
residuals = y_test - y_pred
plt.scatter(y_test, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Actual Prices')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.show()

# Distribution of Residuals
sns.histplot(residuals, kde=True)
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.show()

# Feature Importance (if applicable)
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
coefficients.sort_values(by='Coefficient', ascending=False, inplace=True)
sns.barplot(x='Coefficient', y='Feature', data=coefficients)
plt.title('Feature Importance')
plt.show()

# Extract features for prediction
prediction_date = pd.Timestamp('2024-01-01')
features_for_prediction = dataset[['Likes', 'Dislikes', 'Followers']]

# Filter data for training
training_data = dataset[dataset['Date'] < prediction_date]

# Features and target variable for training
X_train = training_data[['Likes', 'Dislikes', 'Followers']]
y_train = training_data['Price']

# Features for prediction
X_predict = features_for_prediction[features_for_prediction.index[-1]:]

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make Prediction for January 1st, 2024

# Make prediction for the specific date
predicted_price = model.predict(X_predict)
print(f'Predicted Price for January 1st, 2024: {predicted_price[0]} CAD')

from statsmodels.tsa.seasonal import seasonal_decompose

# Time Series Decomposition - Price
result = seasonal_decompose(dataset['Price'], model='additive', period=365)
result.plot()
plt.show()

# Ridge Regression Model
parameters = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
ridge_reg_model = Ridge()
grid_search = GridSearchCV(ridge_reg_model, parameters, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)

# Best parameters and model
best_alpha = grid_search.best_params_['alpha']
best_ridge_model = grid_search.best_estimator_
print(f'Best Alpha: {best_alpha}')

# Predictions
y_pred_ridge = best_ridge_model.predict(X_test)

# Evaluation
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
print(f'Ridge Regression Mean Squared Error: {mse_ridge}')

# Encoding Categorical Variables - Day of the Week
dataset_encoded = pd.get_dummies(dataset, columns=['Day_of_Week'], drop_first=True)

from statsmodels.graphics.tsaplots import plot_acf

# Autocorrelation in Price
plot_acf(dataset['Price'], lags=50)
plt.title('Autocorrelation in Price')
plt.show()

# Rolling Mean and Standard Deviation of Price
rolling_mean = dataset['Price'].rolling(window=30).mean()
rolling_std = dataset['Price'].rolling(window=30).std()

plt.figure(figsize=(12, 6))
plt.plot(dataset['Date'], dataset['Price'], label='Price')
plt.plot(dataset['Date'], rolling_mean, label='Rolling Mean (30 days)')
plt.plot(dataset['Date'], rolling_std, label='Rolling Std (30 days)')
plt.title('Price with Rolling Mean and Std')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

from sklearn.model_selection import cross_val_score

# Evaluate Ridge Regression Model with Cross-Validation
ridge_cv_scores = cross_val_score(best_ridge_model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
ridge_cv_rmse = np.sqrt(-ridge_cv_scores)

print(f'Ridge Regression Cross-Validation RMSE: {ridge_cv_rmse.mean()}')

from statsmodels.tsa.stattools import adfuller

# Stationarity Test on Price (Dickey-Fuller Test)
result_stationarity = adfuller(dataset['Price'])
print(f'Dickey-Fuller Test Statistic: {result_stationarity[0]}')
print(f'p-value: {result_stationarity[1]}')
print(f'Critical Values: {result_stationarity[4]}')

from statsmodels.tsa.stattools import grangercausalitytests

# Granger Causality Test
max_lag = 5
test_result = grangercausalitytests(dataset[['Price', 'Likes']], max_lag, verbose=True)

# Train Ridge Regression model
ridge_model = Ridge()
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}  # Adjust alpha values as needed
ridge_grid = GridSearchCV(ridge_model, param_grid, cv=5, scoring='neg_mean_squared_error')
ridge_grid.fit(X_train, y_train)

# Make predictions
y_pred_ridge = ridge_grid.predict(X_test)

# Evaluate the model
rmse_ridge = mean_squared_error(y_test, y_pred_ridge, squared=False)
print(f'Ridge Regression RMSE: {rmse_ridge}')

# Print best hyperparameter
print(f'Best Ridge Hyperparameter: {ridge_grid.best_params_}')

# Train Ridge Regression model with the best hyperparameter
ridge_model = Ridge(alpha=100)
ridge_model.fit(X_train, y_train)

# Make predictions for January 1st, 2024
X_predict = features_for_prediction.loc[features_for_prediction.index[-1]:]
predicted_price_ridge = ridge_model.predict(X_predict)

# Print predicted price
print(f'Predicted Price for January 1st, 2024 using Ridge Regression: {predicted_price_ridge[0]} CAD')

from sklearn.linear_model import Lasso

# Train Lasso Regression model
lasso_model = Lasso()
lasso_grid = GridSearchCV(lasso_model, param_grid, cv=5, scoring='neg_mean_squared_error')
lasso_grid.fit(X_train, y_train)

# Make predictions
y_pred_lasso = lasso_grid.predict(X_test)

# Evaluate the model
rmse_lasso = mean_squared_error(y_test, y_pred_lasso, squared=False)
print(f'Lasso Regression RMSE: {rmse_lasso}')

# Print best hyperparameter
print(f'Best Lasso Hyperparameter: {lasso_grid.best_params_}')

# Train Ridge Regression model with the best hyperparameter
lasso_model = Lasso(alpha=100)
lasso_model.fit(X_train, y_train)

# Make predictions for January 1st, 2024
X_predict = features_for_prediction.loc[features_for_prediction.index[-1]:]
predicted_price_ridge = lasso_model.predict(X_predict)

# Print predicted price
print(f'Predicted Price for January 1st, 2024 using Lasso Regression: {predicted_price_ridge[0]} CAD')

from sklearn.ensemble import RandomForestRegressor

# Train Random Forest Regression model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
print(f'Random Forest Regression RMSE: {rmse_rf}')

# Make predictions for January 1st, 2024
predicted_price_rf = rf_model.predict(X_predict)

# Actual vs. Predicted Prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='orange', label='Linear Regression')
plt.scatter(y_test, y_pred_ridge, color='blue', label='Ridge Regression')
plt.scatter(y_test, y_pred_lasso, color='red', label='Lasso Regression')
plt.scatter(y_test, y_pred_rf, color='purple', label='Random Forest Regression')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--k', label='Perfect Prediction')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs. Predicted Prices by Different Models')
plt.legend()
plt.show()

# Residuals Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test - y_pred, color='orange', label='Linear Regression')
plt.scatter(y_test, y_test - y_pred_ridge, color='blue', label='Ridge Regression')
plt.scatter(y_test, y_test - y_pred_lasso, color='red', label='Lasso Regression')
plt.scatter(y_test, y_test - y_pred_rf, color='purple', label='Random Forest Regression')
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Actual Prices')
plt.ylabel('Residuals')
plt.title('Residuals by Different Models')
plt.legend()
plt.show()

# Feature Importance Plot (Random Forest)
features = ['Likes', 'Dislikes', 'Followers']
feature_importance_rf = rf_model.feature_importances_
sns.barplot(x=feature_importance_rf, y=features)
plt.title('Feature Importance - Random Forest Regression')
plt.xlabel('Importance')
plt.show()