import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy import stats
from PIL import Image
#import matplotlib.pyplot as plt
#import seaborn as sns


pickle_in = open('kmeans_model.pk2', 'rb')
loaded_model = pickle.load(pickle_in)


def predict_value(Age, Annual_Income, Spending_Score):
    prediction = loaded_model.predict([[Age,Annual_Income,Spending_Score]])
    #print(prediction)
    return prediction



def main():
    st.title("Customer Segmentation")
    

    Age = st.text_input("Enter your Age")
    Annual_Income = st.text_input("Enter your Income")
    Spending_Score = st.text_input("Enter your Spending Score")

    result = ""
    if st.button("Predict"):
        result=predict_value(Age,Annual_Income,Spending_Score)
    st.success("Predicted Clustered Label: {}".format(result))



if __name__ == "__main__":
    main()
