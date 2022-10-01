# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 16:21:55 2022

@author: SHES
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 16:21:55 2022

@author: SHES
"""
import streamlit as st
import seaborn as sns
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
 

header = st.container()
dataset = st.container()
feature = st.container()
modeltraining = st.container()


with header:
    st.title("Data Analysis in Physics and Astronomy")
    st.text("In this study, we try to make a model to distinguish between cancerous and ")
    st.text("noncancerous cells based on the Breast Cancer Wisconsin (Diagnostic) ")
    st.text("dataset on Kaggle.")
    
with dataset:
    
    st.header("two data sets")

    
    df_cancer = pd.read_csv("cancer_data.csv")
    st.subheader("cancer distribution data set 2020 in India")
    st.write(df_cancer.head(6))
    df_cancer.sort_values(by=["Deaths Number"],ascending=False, inplace=True)
    df_cancer.set_index("cancer_type",inplace=True)
    cancers = list(df_cancer.index[0:9])
    top_5_death = df_cancer.loc[cancers , "Deaths Number"]
    st.text(" ”WHO” cancer statistics dataset of India in 2020, which is available on Kaggle")
    fig = plt.figure(figsize=(10, 4))
    top_5_death.plot(kind="bar", figsize=(10, 6) , rot=40,color=["red",'darkgreen','darkgreen','darkgreen','darkgreen','darkgreen','darkgreen','darkgreen','darkgreen','darkgreen'] )
    st.bar_chart(top_5_death)
    
    df = pd.read_csv("data.csv")
    st.subheader("breast cancer data set")
    st.write(df.head(6)) 
    mean_features3 = list(df.columns[2:12])          ### making list of the columns names
    se_features3 = list(df.columns[12:22])
    worst_features3 = list(df.columns[22:32])
    dia = df.diagnosis                                        # M or B 
    list = ['Unnamed: 32','id','diagnosis']                   # list of unnecessary data
    fea = df.drop(list,axis = 1 )                             # drop the coloumns , and assign to fea  
    sns.set(style="darkgrid")
    fig = plt.figure(figsize=(10, 4))
    ax = sns.countplot(x=df.diagnosis,label="count")
    st.pyplot(fig)


with feature:
    
    st.header("features")
    st.text("after processing our data now we can take a look at it:") 

    df["diagnosis"] = df["diagnosis"].map({"M":1,"B":0})          ## converting M and b to 1 and 0
    
    st.text("Violin plot, It shows the distribution of quantitative data across several levels of ")
    st.text("one (or more) categorical variables")
    y = dia
    x = fea
    data_st = (x - x.mean()) / (x.std())              # standardization
    data = pd.concat([dia,data_st.iloc[:,0:10]],axis=1)
    data = pd.melt(data,id_vars="diagnosis",var_name="features",value_name='value')
    fig=plt.figure(figsize=(10,10))
    pal_col = {"M": "r", "B": "g"}
    sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True,palette=pal_col)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    st.caption("violinplot of all the mean features")
    
    
    y = dia
    x = fea
    data_st = (x - x.mean()) / (x.std())              # standardization
    data = pd.concat([dia,data_st.iloc[:,10:20]],axis=1)
    data = pd.melt(data,id_vars="diagnosis",var_name="features",value_name='value')
    fig=plt.figure(figsize=(10,10))
    pal_col = {"M": "r", "B": "g"}
    sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True,palette=pal_col)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    st.caption("violinplot of all the standard error features")
    
    y = dia
    x = fea
    data_st = (x - x.mean()) / (x.std())              # standardization
    data = pd.concat([dia,data_st.iloc[:,20:31]],axis=1)
    data = pd.melt(data,id_vars="diagnosis",var_name="features",value_name='value')
    fig = plt.figure(figsize=(10,10))
    pal_col = {"M": "r", "B": "g"}
    sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True,palette=pal_col)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    st.caption("violinplot of all the worst features")   
    
    ########
    st.text("here we can take a look at correlation heatmap to choose better features.")
    
    fig = plt.figure(figsize=(12,12))
    sns.heatmap(df[mean_features3].corr(), annot=True)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.text("you can try it yourself!")
    sel_col, disp_col = st.columns(2)
    
    first_feature = sel_col.text_input("what should be used as first feature",'concave points_mean')
    second_feature = sel_col.text_input("what should be used as second feature",'area_mean')
    
    ax = sns.jointplot(data = data_st,x= first_feature,y=second_feature, kind="reg",color ="darkcyan")
    st.pyplot(ax)
    #######
    fig = plt.figure(figsize=(12,12))
    sns.heatmap(df[se_features3].corr(), annot=True)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.text("you can try it yourself!")
    sel_col, disp_col = st.columns(2)
    
    first_feature = sel_col.text_input("what should be used as first feature",'concave points_se')
    second_feature = sel_col.text_input("what should be used as second feature",'area_se')
    
    ax = sns.jointplot(data = data_st,x= first_feature,y=second_feature, kind="reg",color ="darkcyan")
    st.pyplot(ax)
    #######
    fig = plt.figure(figsize=(12,12))
    sns.heatmap(df[worst_features3].corr(), annot=True)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.text("you can try it yourself!")
    sel_col, disp_col = st.columns(2)
    
    first_feature = sel_col.text_input("what should be used as first feature",'concave points_worst')
    second_feature = sel_col.text_input("what should be used as second feature",'area_worst')
    
    
    ax = sns.jointplot(data = data_st,x= first_feature,y=second_feature, kind="reg",color ="darkcyan")
    st.pyplot(ax)
    
    
    ######
    
    
with modeltraining:
    st.header("Time to train model")
    st.header("Here we can try different features and parameters for our model!") 
    sel_col, disp_col = st.columns(2)
    max_depth = sel_col.slider("What should be the max_depth of the model?", min_value=10, max_value=100, value=20, step=10)
    n_estimators = sel_col.selectbox("how many trees ?", options=[100,200,300],index=0)
    input_feature = sel_col.text_input("what should be used as input feature","radius_mean")
    
    df = pd.read_csv("data.csv")
    reg = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
      
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
