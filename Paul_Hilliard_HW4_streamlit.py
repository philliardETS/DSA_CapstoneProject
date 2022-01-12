import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# PAUL HILLIARD - HOMEWORK 4 DUE 11/26/21
# USING PROVIDED CSV FILE OF DATA TO PRODUCE VISUALIZATIONS - ALSO IN VOILA

#import necessary libraries
#import matplotlib.pyplot as plt
import plotly.graph_objs as go

#load in the science data using pandas
data = pd.read_csv("science_response.csv")
df=pd.DataFrame(data)
df

# Get some statistics about the first few items stored as 0=incorrect 1=correct so we can display it in some charts
# Broken down by Gender

item1_1=sum(df.item1)
item1_0=sum(df.item1 ==0)

item1_0_M=sum((df.item1==0)&(df.gender=='Male'))
item1_0_F=sum((df.item1==0)&(df.gender=='Female'))

item1_1_M=sum((df.item1)&(df.gender=='Male'))
item1_1_F=sum((df.item1)&(df.gender=='Female'))
print(item1_0)
print(item1_1)
print(item1_0_M)
print(item1_0_F)
print(item1_1_M)
print(item1_1_F)

item2_1=sum(df.item2)
item2_0=sum(df.item2 ==0)

item2_0_M=sum((df.item2==0)&(df.gender=='Male'))
item2_0_F=sum((df.item2==0)&(df.gender=='Female'))

item2_1_M=sum((df.item2)&(df.gender=='Male'))
item2_1_F=sum((df.item2)&(df.gender=='Female'))
print(item2_0)
print(item2_1)
print(item2_0_M)
print(item2_0_F)
print(item2_1_M)
print(item2_1_F)

# Put it in a dataframe called dfitems
dataitems={'value': ['incorrect','correct'],
           'item1': [item1_0,item1_1],
           'item1_M': [item1_0_M,item1_1_M],
           'item1_F': [item1_0_F,item1_1_F],
           'item2': [item2_0,item2_1],
           'item2_M': [item2_0_M,item2_1_M],
           'item2_F': [item2_0_F,item2_1_F]}


dfitems=pd.DataFrame(dataitems)
dfitems
	   
# Do a pie chart showing the percent corect for item 1 Overall
fig=px.pie(dfitems,values='item1',names='value',title="Item 1 - Overall")
fig.show()
	   
# Males
fig=px.pie(dfitems,values='item1_M',names='value',title="Item 1 - Males")
fig.show()
	   
#Females
fig=px.pie(dfitems,values='item1_F',names='value',title="Item 1 - Females")
fig.show()
	   
# I wanted to have data more in a summary style.  I believe this can be done using
# a pivot table in Python but I used Excel on the .csv file and created a new datafile

#load in the formatted science data using pandas.  A breakdown of total scores by Gender
data = pd.read_csv("science_response_sum_score.csv")
dfsum=pd.DataFrame(data)
dfsum

#pie chart of sum scores, overall
fig=px.pie(dfsum,values='Total',names='sum_score',title="Breakdown of Sum Scores - Overall")
fig.show()

#pie chart of sum scores, for Males
fig=px.pie(dfsum,values='Male',names='sum_score',title="Breakdown of Sum Scores - Male")
fig.show()

#pie chart of sum scores, for Females
fig=px.pie(dfsum,values='Female',names='sum_score',title="Breakdown of Sum Scores - Female")
fig.show()

#as a histogram
fig=px.bar(dfsum,x='sum_score',y='Total',title="Breakdown of Sum Scores - Overall")
fig.show()

# Stacked with Males and Females displayed.  I tried to do a radio button for Voila but just kept getting errors
fig=px.bar(dfsum,x='sum_score',y=['Male','Female'],title="Breakdown of Sum Scores - by Gender")
fig.show()
