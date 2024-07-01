# -*- coding: utf-8 -*-

!pip install pyspark

# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when

# Initialize Spark session
spark = SparkSession.builder.appName('GoldPricePrediction').getOrCreate()

# Load the data into a Spark DataFrame
gold_data = spark.read.csv('/content/goldprices.csv', header=True, inferSchema=True)

# Show first 5 rows
gold_data.show(5)

# Show last 5 rows
gold_data.orderBy(col("Date").desc()).show(5)

# Number of rows and columns
print((gold_data.count(), len(gold_data.columns)))

# Get some basic information about the data
gold_data.printSchema()

# Checking the number of missing values
gold_data.select([count(when(col(c).isNull(), c)).alias(c) for c in gold_data.columns]).show()

# Convert to Pandas DataFrame for further processing
gold_data_pd = gold_data.toPandas()

# Drop the 'Date' column for correlation calculation
gold_data_pd = gold_data_pd.drop(['Date'], axis=1)

# Calculate correlation
correlation = gold_data_pd.corr()

# Constructing a heatmap to understand the correlation
plt.figure(figsize=(8, 8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='Blues')

# Correlation values of GLD
print(correlation['GLD'])

# Checking the distribution of the GLD Price
sns.distplot(gold_data_pd['GLD'], color='green')

# Splitting the Features and Target
X = gold_data_pd.drop(['GLD'], axis=1)
Y = gold_data_pd['GLD']

print(X)
print(Y)

# Splitting into Training data and Test Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Model Training: Random Forest Regressor
regressor = RandomForestRegressor(n_estimators=100)

# Training the model
regressor.fit(X_train, Y_train)

# Model Evaluation
# Prediction on Test Data
test_data_prediction = regressor.predict(X_test)
print(test_data_prediction)

# R squared error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared error: ", error_score)

# Compare the Actual Values and Predicted Values in a Plot
Y_test = list(Y_test)
plt.plot(Y_test, color='blue', label='Actual Value')
plt.plot(test_data_prediction, color='green', label='Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()
