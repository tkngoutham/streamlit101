#In task 1, you were required to show the first 10 rows from the candydData.csv file using Streamlit UI.

# CopyRights : https://www.kaggle.com/code/gcdatkin/candy-bar-prediction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("./datasets/candyData.csv") # Reading dataset
# Solution
st.title("Candy Bar Dataset") # It will add the title for the dataset
# The following line will display the text as sub heading
st.subheader("This model will predict whether the candy bar is a bar or not")
st.table(data.head(11)) # This line displays the dataset from 0-10 rows