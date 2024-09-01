#In task 2, you were required to display the heatmap for the candydata.csv dataset. The following code displays the heatmap on Streamlit UI.
# CopyRights : https://www.kaggle.com/code/gcdatkin/candy-bar-prediction
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("./datasets/candyData.csv") # Reading dataset
names = data['competitorname']
data.drop('competitorname', axis=1, inplace=True)
figure = plt.figure(figsize=(12, 10)) # Building figure for heat map
sns.heatmap(data.corr(), annot=True, vmin=-1, vmax=1) # Heat map
st.pyplot(figure) # This line will display the heat map on streamlit UI