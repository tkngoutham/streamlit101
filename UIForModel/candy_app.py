# CopyRights : https://www.kaggle.com/code/gcdatkin/candy-bar-prediction
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("./datasets/candyData.csv") # Reading the dataset
names = data['competitorname']
data.drop('competitorname', axis=1, inplace=True)
plt.figure(figsize=(12, 10)) # Building figure for heat map
sns.heatmap(data.corr(), annot=True, vmin=-1, vmax=1) # Heat map
data.isnull().sum()
y = data['bar'] # Declaring bar attribute as a class
X = data.drop('bar', axis=1)
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
model = LogisticRegression()
model.fit(X_train, y_train) # Predicting the model
print(f"Model Accuracy: {model.score(X_test, y_test)}") # Printing accuracy
# Write your code here
