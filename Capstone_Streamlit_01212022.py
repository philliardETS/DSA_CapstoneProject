# PAUL HILLIARD - CAPSTONE PROJECT
# DATA SCIENCE ACADEMY
# 2021 - 2022

#import necessary libraries

import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import datetime
import json
import plotly.io as pio
import matplotlib.pyplot as plt
import plotly.graph_objs as go
pio.templates.default="ggplot2"
	
#load in the science data using pandas
data = pd.read_csv("data_capstone_dsa2021_2022.csv")
df=pd.DataFrame(data)
df

# sum_score, gender, home_computer are clean.  
#bin the Ages and the Performance
df['age_group'] = pd.qcut(df['age'], 5, 
    labels=['Age Group1: < 28', 'Age Group2: 28-31', 'Age Group3: 32-36', 'Age Group4: 37-44', 'Age Group5: > 45'])
df['ability'] = pd.qcut(df['sum_score'], 3, labels=['Low', 'Mid', 'High'])
#recode the gender to 1=Males, 2=Females, to use in scatterplots
#and home_computer to 1=YES, 2=NO, to use in scatterplots
df.loc[df['gender'] == 'Female', 'gender_coded'] = 1
df.loc[df['gender'] == 'Male', 'gender_coded'] = 2
df.loc[df['home_computer'] == 'No', 'home_computer_coded'] = 1
df.loc[df['home_computer'] == 'Yes', 'home_computer_coded'] = 2

#take a look at the file
df.head(20)
df.tail(20)

#Pie Chart of some things
#1-3
gendercounts=pd.DataFrame(df.groupby('gender').count())
agecounts=pd.DataFrame(df.groupby('age_group').count())
homecompcounts=pd.DataFrame(df.groupby('home_computer').count())
genderlevs=["Female","Male"]
agelevs=["Age Group1: < 28", "Age Group2: 28-31", "Age Group3: 32-36", "Age Group4: 37-44", "Age Group5: > 45"]
homecomplevs=["Home Computer: No","Home Computer: Yes"]

fig=px.pie(gendercounts, values='sum_score', names=genderlevs)
st.plotly_chart(fig)
fig=px.pie(agecounts, values='sum_score', names=agelevs)
st.plotly_chart(fig)
fig=px.pie(homecompcounts, values='sum_score', names=homecomplevs)
st.plotly_chart(fig)

