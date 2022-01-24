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
df = pd.read_csv("data/data_capstone_dsa2021_2022.csv")
df=pd.DataFrame(data)
#df

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
#df.head(20)
#df.tail(20)

#Pie Chart of some things
#1-3

st.title("Data Science Academy Capstone Project")
st.title("Paul Hilliard - 2021-2022")

st.markdown("---")
st.markdown("## Description of Data")
st.markdown("The datafile consists of a 20-item assessment, with item level timing and score performance.  Total assessment timing and score are also provided.  ")

st.markdown("Categorical record variables include Gender, Age, whether the assessment was taken on their Home Computer, and their Location." )

st.markdown("---")

gendercounts=pd.DataFrame(df.groupby('gender').count())
agecounts=pd.DataFrame(df.groupby('age_group').count())
homecompcounts=pd.DataFrame(df.groupby('home_computer').count())
genderlevs=["Female","Male"]
agelevs=["Age Group1: < 28", "Age Group2: 28-31", "Age Group3: 32-36", "Age Group4: 37-44", "Age Group5: > 45"]
homecomplevs=["Home Computer: No","Home Computer: Yes"]

st.title("Some Basic Categorical Displays")
st.markdown("## Pie Chart - Proportion of Records - by Gender")
fig=px.pie(gendercounts, values='sum_score', names=genderlevs)
st.plotly_chart(fig)
st.markdown("## Pie Chart - Proportion of Records - by Age Split into 5 Approximately Equal Groups")
fig=px.pie(agecounts, values='sum_score', category_orders={"age_group":["Age Group1: < 28", "Age Group2: 28-31", "Age Group3: 32-36", "Age Group4: 37-44", "Age Group5: > 45"]},names=agelevs)
st.plotly_chart(fig)
st.markdown("## Pie Chart - Proportion of Records - by Home Computer Use")
fig=px.pie(homecompcounts, values='sum_score', names=homecomplevs)
st.plotly_chart(fig)

# Time Distributions
timingtotfreqs_all=pd.DataFrame(df['rt_total'])

#By Gender
timingtotfreqs_male=df[df['gender'] =="Male"]
timingtotfreqs_male=timingtotfreqs_male["rt_total"]
timingtotfreqs_female=df[df['gender'] =="Female"]
timingtotfreqs_female=timingtotfreqs_female["rt_total"]

#By Ability
timingtotfreqs_lowabil=df[df['ability'] == "Low"]
timingtotfreqs_midabil=df[df['ability'] == "Mid"]
timingtotfreqs_highabil=df[df['ability'] == "High"]
timingtotfreqs_lowabil=timingtotfreqs_lowabil["rt_total"]
timingtotfreqs_midabil=timingtotfreqs_midabil["rt_total"]
timingtotfreqs_highabil=timingtotfreqs_highabil["rt_total"]

#By Home Computer Use
timingtotfreqs_HomeY=df[df['home_computer'] == "Yes"]
timingtotfreqs_HomeN=df[df['home_computer'] == "No"]
timingtotfreqs_HomeY=timingtotfreqs_HomeY["rt_total"]
timingtotfreqs_HomeN=timingtotfreqs_HomeN["rt_total"]

# SUM SCORE Distributions
scoretotfreqs_all=pd.DataFrame(df['sum_score'])

#By Gender
scoretotfreqs_male=df[df['gender'] =="Male"]
scoretotfreqs_male=scoretotfreqs_male["sum_score"]
scoretotfreqs_female=df[df['gender'] =="Female"]
scoretotfreqs_female=scoretotfreqs_female["sum_score"]

#By HomeComputer
scoretotfreqs_HomeY=df[df['home_computer'] =="Yes"]
scoretotfreqs_HomeY=scoretotfreqs_HomeY["sum_score"]
scoretotfreqs_HomeN=df[df['home_computer'] =="No"]
scoretotfreqs_HomeN=scoretotfreqs_HomeN["sum_score"]

#Overall Timing
#4
fig = px.histogram(df, x="rt_total", nbins=50)

fig.update_xaxes(range=[0,5000])
st.plotly_chart(fig)


#Overall Score
#5
fig = px.histogram(df, x="sum_score", nbins=20)

fig.update_xaxes(range=[0,21])
st.plotly_chart(fig)

#This prints a nice overlap of 2 distributions
#6

fig = go.Figure()
fig.add_trace(go.Histogram(x=timingtotfreqs_male,nbinsx=50,name='Total Time: Males'))
fig.add_trace(go.Histogram(x=timingtotfreqs_female,nbinsx=50,name='Total Time: Females'))


# Overlay both histograms
fig.update_layout(barmode='overlay')
# Reduce opacity to see both histograms
fig.update_traces(opacity=0.50)
st.plotly_chart(fig)

#7
fig = go.Figure()
fig.add_trace(go.Histogram(x=scoretotfreqs_female,name='Total Score: Females'))
fig.add_trace(go.Histogram(x=scoretotfreqs_male,name='Total Score: Males'))

# Overlay both histograms
fig.update_layout(barmode='overlay')
# Reduce opacity to see both histograms
fig.update_traces(opacity=0.50)
st.plotly_chart(fig)


#This prints a nice overlap of 2 distributions
#8

fig = go.Figure()
fig.add_trace(go.Histogram(x=timingtotfreqs_HomeY,nbinsx=50,name='Total Time: Home Computer (Yes)'))
fig.add_trace(go.Histogram(x=timingtotfreqs_HomeN,nbinsx=50,name='Total Time: Home Computer (No)'))


# Overlay both histograms
fig.update_layout(barmode='overlay')
# Reduce opacity to see both histograms
fig.update_traces(opacity=0.50)
st.plotly_chart(fig)



#from plotly.offline import plot

#layout = {}
#traces = []

#traces.append({'x': timingtotfreqs_female, 'name': 'Overall Timing: Females', 'opacity': 0.25})
#traces.append({'x': timingtotfreqs_male, 'name': 'Overall Timing: Males', 'opacity': 0.25})

# For each trace, add elements which are common to both.
#for t in traces:
#   t.update({'type': 'histogram',
  #            'histfunc': 'count',
    #          'nbinsx': 100})

#layout['barmode'] = 'overlay'

#plot({'data': traces, 'layout': layout})
#st.plotly_chart(plot)

#This prints a nice overlap of 2 distributions
#7

#layout = {}
#traces = []

#traces.append({'x': scoretotfreqs_female, 'name': 'Sum Score Distribution: Females', 'opacity': 0.25})
#traces.append({'x': scoretotfreqs_male, 'name': 'Sum Score Distribution: Males', 'opacity': 0.25})

# For each trace, add elements which are common to both.
#for t in traces:
  #  t.update({'type': 'histogram',
    #          'histfunc': 'count',
      #        'nbinsx': 20})

#layout['barmode'] = 'overlay'

#plot({'data': traces, 'layout': layout})
#st.plotly_chart(plot)

#This prints a nice overlap of 2 distributions
#8
#from plotly.offline import plot

#layout = {}
#traces = []

#traces.append({'x': timingtotfreqs_HomeY, 'name': 'Overall Timing: Home Computer - Yes', 'opacity': 0.25})
#traces.append({'x': timingtotfreqs_HomeN, 'name': 'Overall Timing: Home Computer - No', 'opacity': 0.25})

# For each trace, add elements which are common to both.
#for t in traces:
 #   t.update({'type': 'histogram',
  #            'histfunc': 'count',
   #           'nbinsx': 100})

#layout['barmode'] = 'overlay'

#plot({'data': traces, 'layout': layout})
#st.plotly_chart(plot)

#9
TimePlot = px.scatter(df, x="rt_total", y="sum_score", size="sum_score", color="ability", hover_name="ability", log_x=False, size_max=15)
st.plotly_chart(TimePlot)

#10
TimePlot = px.scatter(df, x="rt_total", y="age_group", size="sum_score", color="age_group", hover_name="ability", log_x=False, size_max=25)
st.plotly_chart(TimePlot)

#11
TimePlot = px.scatter(df, x="rt_total", y="gender_coded", size="sum_score", color="gender_coded", hover_name="gender", log_x=False, size_max=75)
st.plotly_chart(TimePlot)

#Means for all Timing and Scored Item variables
#df.groupby('group_column')['sum_column'].sum() 
#this gives us the # correct for gs_1

allmeans=pd.DataFrame(df.mean())
allmeansT=pd.DataFrame(allmeans.T)
allgendermeans=pd.DataFrame(df.groupby('gender').mean())
allagemeans=pd.DataFrame(df.groupby('age_group').mean())
allabilmeans=pd.DataFrame(df.groupby('ability').mean())
allhomecompmeans=pd.DataFrame(df.groupby('home_computer').mean())

print(allmeansT)
print(allgendermeans)
print(allagemeans)
print(allabilmeans)
print(allhomecompmeans)

#This is just timing and scores overall
#allmeansT_time=allmeansT.iloc[:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]
allmeansT_time=allmeansT.loc[:,allmeansT.columns.str.startswith('rt_gs')]
allmeansT_scores=allmeansT.loc[:,allmeansT.columns.str.startswith('gs')]
#allmeansT_time
#allmeansT_scores

#This is all items, timing and scoring
female=allgendermeans.head(1)
male=allgendermeans.tail(1)
allmeansT=allmeansT.head(1)
allmeansT_time=allmeansT_time.head(1)
allmeansT_scores=allmeansT_scores.head(1)
female=female.to_numpy()
male=male.to_numpy()
allmeansT=allmeansT.to_numpy()
allmeansT_time=allmeansT_time.to_numpy()
allmeansT_scores=allmeansT_scores.to_numpy()
female=female.flatten()
male=male.flatten()
allmeansT=allmeansT.flatten()
allmeansT_time=allmeansT_time.flatten()
allmeansT_scores=allmeansT_scores.flatten()
print(female)
print(male)
print(allmeansT)
print(allmeansT_time)
print(allmeansT_scores)


#Try to do a histogram of all timing values overall
#12
x_axis_label=  ['I2','I3','I4','I5','I6','I7','I8','I9','I10',
                'I11','I12','I13','I14','I15','I16','I17','I18','I19','I20']
x_axis = np.arange(len(x_axis_label))
# Multi bar Chart
print(x_axis)
print(allmeansT_time)

plt.bar(x_axis, allmeansT_time, width=0.8, label = 'Item Timing: All Records')
#plt.bar(x_axis +0.2, male, width=0.4, label = 'Male')

# Xticks

plt.xticks(x_axis, x_axis_label)

# Add legend

plt.legend()
# Display

st.pyplot(plt)
plt.clf()
#Items 7, 9 and 13 had highest time

#Try to do a histogram of all item performance
#13
x_axis_label=  ['I1','I2','I3','I4','I5','I6','I7','I8','I9','I10',
                'I11','I12','I13','I14','I15','I16','I17','I18','I19','I20']
x_axis = np.arange(len(x_axis_label))
# Multi bar Chart
print(x_axis)
print(allmeansT_scores)

plt.bar(x_axis, allmeansT_scores, width=0.8, label = 'Item Difficulty: All Records')
#plt.bar(x_axis +0.2, male, width=0.4, label = 'Male')

# Xticks

plt.xticks(x_axis, x_axis_label)

# Add legend

plt.legend()
# Display

st.pyplot(plt)
plt.clf()
#Items 7, 9 and 13 had highest time

#Examine Items 7, 9, and 13 in more detail  
#Break down timing by gender and ability level
#Break down performance by gender and age
#allgendermeans
#allabilmeans
#allagemeans

gender_T_I7_I9_I13=allgendermeans[['rt_gs_7','rt_gs_9','rt_gs_13']]
abil_T_I7_I9_I13=allabilmeans[['rt_gs_7','rt_gs_9','rt_gs_13']]

gender_S_I7_I9_I13=allgendermeans[['gs_7','gs_9','gs_13']]
age_S_I7_I9_I13=allagemeans[['gs_7','gs_9','gs_13']]

#next convert so I can display side by side histograms.  It needs to be a flat 1-row array

female_T_I7_I9_I13=gender_T_I7_I9_I13.head(1)
female_T_I7_I9_I13=female_T_I7_I9_I13.to_numpy()
female_T_I7_I9_I13=female_T_I7_I9_I13.flatten()
male_T_I7_I9_I13=gender_T_I7_I9_I13.tail(1)
male_T_I7_I9_I13=male_T_I7_I9_I13.to_numpy()
male_T_I7_I9_I13=male_T_I7_I9_I13.flatten()

lowabil_T_I7_I9_I13=abil_T_I7_I9_I13.iloc[[0]]
lowabil_T_I7_I9_I13=lowabil_T_I7_I9_I13.to_numpy()
lowabil_T_I7_I9_I13=lowabil_T_I7_I9_I13.flatten()
midabil_T_I7_I9_I13=abil_T_I7_I9_I13.iloc[[1]]
midabil_T_I7_I9_I13=midabil_T_I7_I9_I13.to_numpy()
midabil_T_I7_I9_I13=midabil_T_I7_I9_I13.flatten()
higabil_T_I7_I9_I13=abil_T_I7_I9_I13.iloc[[2]]
higabil_T_I7_I9_I13=higabil_T_I7_I9_I13.to_numpy()
higabil_T_I7_I9_I13=higabil_T_I7_I9_I13.flatten()

#SCORE PERFORMANCE
female_S_I7_I9_I13=gender_S_I7_I9_I13.head(1)
female_S_I7_I9_I13=female_S_I7_I9_I13.to_numpy()
female_S_I7_I9_I13=female_S_I7_I9_I13.flatten()
male_S_I7_I9_I13=gender_S_I7_I9_I13.tail(1)
male_S_I7_I9_I13=male_S_I7_I9_I13.to_numpy()
male_S_I7_I9_I13=male_S_I7_I9_I13.flatten()

age_1_S_I7_I9_I13=age_S_I7_I9_I13.iloc[[0]]
age_1_S_I7_I9_I13=age_1_S_I7_I9_I13.to_numpy()
age_1_S_I7_I9_I13=age_1_S_I7_I9_I13.flatten()
age_2_S_I7_I9_I13=age_S_I7_I9_I13.iloc[[1]]
age_2_S_I7_I9_I13=age_2_S_I7_I9_I13.to_numpy()
age_2_S_I7_I9_I13=age_2_S_I7_I9_I13.flatten()
age_3_S_I7_I9_I13=age_S_I7_I9_I13.iloc[[2]]
age_3_S_I7_I9_I13=age_3_S_I7_I9_I13.to_numpy()
age_3_S_I7_I9_I13=age_3_S_I7_I9_I13.flatten()
age_4_S_I7_I9_I13=age_S_I7_I9_I13.iloc[[3]]
age_4_S_I7_I9_I13=age_4_S_I7_I9_I13.to_numpy()
age_4_S_I7_I9_I13=age_4_S_I7_I9_I13.flatten()
age_5_S_I7_I9_I13=age_S_I7_I9_I13.iloc[[4]]
age_5_S_I7_I9_I13=age_5_S_I7_I9_I13.to_numpy()
age_5_S_I7_I9_I13=age_5_S_I7_I9_I13.flatten()

#Try to do a histogram of all timing values overall
#14
x_axis_label=  ['Item 7','Item 9','Item 13']
x_axis = np.arange(len(x_axis_label))

# Multi bar Chart
# Multi bar Chart

plt.bar(x_axis -0.2, female_T_I7_I9_I13, width=0.4, label = 'Female')
plt.bar(x_axis +0.2, male_T_I7_I9_I13, width=0.4, label = 'Male')

# Xticks

plt.xticks(x_axis, x_axis_label)

# Add legend

plt.legend()

# Display

st.pyplot(plt)
plt.clf()
#Try to do a histogram of all score values overall
#15
x_axis_label=  ['Item 7','Item 9','Item 13']
x_axis = np.arange(len(x_axis_label))

# Multi bar Chart
# Multi bar Chart

plt.bar(x_axis -0.2, female_S_I7_I9_I13, width=0.4, label = 'Female')
plt.bar(x_axis +0.2, male_S_I7_I9_I13, width=0.4, label = 'Male')

# Xticks

plt.xticks(x_axis, x_axis_label)

# Add legend

plt.legend()

# Display

st.pyplot(plt)
plt.clf()
#Try to do a histogram of all timing values overall
#16
x_axis_label=  ['Item 7','Item 9','Item 13']
x_axis = np.arange(len(x_axis_label))

# Multi bar Chart
# Multi bar Chart

plt.bar(x_axis -0.3, lowabil_T_I7_I9_I13, width=0.3, label = 'Low')
plt.bar(x_axis +0.0, midabil_T_I7_I9_I13, width=0.3, label = 'Mid')
plt.bar(x_axis +0.3, higabil_T_I7_I9_I13, width=0.3, label = 'High')


# Xticks

plt.xticks(x_axis, x_axis_label)

# Add legend

plt.legend()

# Display

st.pyplot(plt)
plt.clf()
#Try to do a histogram of all timing values overall
#17
x_axis_label=  ['Item 7','Item 9','Item 13']
x_axis = np.arange(len(x_axis_label))

# Multi bar Chart

plt.bar(x_axis -0.3, age_1_S_I7_I9_I13, width=0.15, label = 'Younger than 28 years')
plt.bar(x_axis -0.15, age_2_S_I7_I9_I13, width=0.15, label = '28 to 31 years')
plt.bar(x_axis +0.0, age_3_S_I7_I9_I13, width=0.15, label = '32 to 36 years')
plt.bar(x_axis +0.15, age_4_S_I7_I9_I13, width=0.15, label = '37 to 44 years')
plt.bar(x_axis +0.3, age_5_S_I7_I9_I13, width=0.15, label = 'Older than 44 years')


# Xticks

plt.xticks(x_axis, x_axis_label)
plt.ylim([0,1])
plt.ylabel('Percent Correct')
# Add legend

plt.legend(loc='lower center')

# Display

st.pyplot(plt)
plt.clf()