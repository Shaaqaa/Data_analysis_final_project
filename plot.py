# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 16:21:55 2022

@author: SHES
"""
import streamlit as st
#import seaborn as sns
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
 

header = st.container()
dataset = st.container()
feature = st.container()
modeltraining = st.container()

with header:
    st.title("Data Analysis in Physics and Astronomy")
    st.text("n this study, we try to make a model to distinguish between cancerous and noncancerous cells based on the Breast Cancer Wisconsin (Diagnostic) dataset on Kaggle.")

with dataset:
    
    st.header("two data sets")

    
    df_cancer = pd.read_csv("cancer_data.csv")
    st.subheader("cancer distribution data set 2020 in India")
    st.write(df_cancer.head(6))
    
    st.text(" ”WHO” cancer statistics dataset of India in 2020, which is available on Kaggle")
    bar_in_cancer = Image.open("cancer_india.png")
    st.image(bar_in_cancer,caption=("top 5 cancer with highest number of in india deaths in 2020"))
    
    df = pd.read_csv("data.csv")
    st.subheader("breast cancer data set")
    st.write(df.head(6))    
    
    
   
with feature:
    st.header("features")  
    st.text("Violin plot, It shows the distribution of quantitative data across several levels of one (or more) categorical variables")
    vi_m = Image.open("vi_mean.png")
    st.image(vi_m,caption=("Violin plot of the  features.(mean features)"))
    vi_se = Image.open("vi_se.png")
    st.image(vi_se,caption=("Violin plot of the  features.(standard Error features)"))
    vi_wo = Image.open("vi_worst.png")
    st.image(vi_wo,caption=("Violin plot of the  features.(worst features)"))
    
    """  
    st.text("correlation heat map.")
    he_me  = Image.open("heat_mean.png")
    st.image(he_me,caption=("heat map correlation of the features (mean features)"))
    he_se  = Image.open("heat_se.png")
    st.image(he_se,caption=("heat map correlation of the features (standard error features)"))
    he_wo  = Image.open("heat_worst.png")
    st.image(he_wo,caption=("heat map correlation of the features (worst features)"))
    
    st.text("Joint plot to observe correlation")
    jo_1  = Image.open("joint_se_1.png")
    st.image(jo_1,caption=("Joint plot"))
    
    jo  = Image.open("joint_se.png")
    st.image(jo,caption=("Joint plot"))"""
    
    
with modeltraining:
    st.header("Time to train model")  
    sel_col, disp_col = st.columns(2)
    max_depth = sel_col.slider("What should be the max_depth of the model?", min_value=10, max_value=100, value=20, step=10)
    n_estimators = sel_col.selectbox("how many trees ?", options=[100,200,300],index=0)
    input_feature = sel_col.text_input("what should be used as input feature","radius_mean")
    
    reg = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
    
    df["diagnosis"] = df["diagnosis"].map({"M":1,"B":0}) 
    dia = df.diagnosis                                        # M or B 
    list = ['Unnamed: 32','id','diagnosis']                   # list of unnecessary data
    fea = df.drop(list,axis = 1 )   
      
    
    
    
    x_train, x_test, y_train, y_test = train_test_split(fea, dia, test_size=0.3, random_state=42)
    
    
    x = x_train[[input_feature]]
    y = y_train
    RF = reg.fit(x,y)
    prediction = RF.predict(x_test[[input_feature]])
    
    
    disp_col.subheader("Mean absoulte error of the model is:")
    disp_col.write(mean_absolute_error(y_test, prediction))
    disp_col.subheader("Mean squared  error of the model is:")
    disp_col.write(mean_squared_error(y_test, prediction))
    disp_col.subheader("R^2 score error of the model is:")
    disp_col.write(r2_score(y_test, prediction))
    
    
    
    
    
    
    
    
    
    
    
    
    
    