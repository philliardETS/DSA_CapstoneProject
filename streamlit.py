# PAUL HILLIARD - HOMEWORK 4 - DUE 11/26/2021

# EXERCISE 1.1
# %load exercises/1.1-subplots_and_basic_plotting.py
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import plotly.express as px

# Try to reproduce the figure shown in images/exercise_1-1.png

# Our data...
x = np.linspace(0, 10, 100)
y1, y2, y3 = np.cos(x), np.cos(x + 1), np.cos(x + 2)
names = ['Signal 1', 'Signal 2', 'Signal 3']

# Can you figure out what to do next to plot x vs y1, y2, and y3 on one figure?

# EXERCISE 1.1
# create three separate subplots, presented one on top of one another.  Provide the color and the label. 
# Remove the values on the vertical and horizontal axes.
# I think there may be a way to loop this rather than produce them separately
fig, (ax1, ax2, ax3)=plt.subplots(3)
ax1.plot(x,y1,color='black')
ax1.set_title('Signal 1')
ax1.set(xticks=[], yticks=[])
ax2.plot(x,y2,color='black')
ax2.set_title('Signal 2')
ax2.set(xticks=[], yticks=[])
ax3.plot(x,y3,color='black')
ax3.set_title('Signal 3')
ax3.set(xticks=[], yticks=[])
plt.show()

# EXERCISE 2.1
# %load exercises/2.1-bar_and_fill_between.py
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)

# Generate data...
y_raw = np.random.randn(1000).cumsum() + 15
x_raw = np.linspace(0, 24, y_raw.size)

# Get averages of every 100 samples...
x_pos = x_raw.reshape(-1, 100).min(axis=1)
y_avg = y_raw.reshape(-1, 100).mean(axis=1)
y_err = y_raw.reshape(-1, 100).ptp(axis=1)

bar_width = x_pos[1] - x_pos[0]

# Make a made up future prediction with a fake confidence
x_pred = np.linspace(0, 30)
y_max_pred = y_avg[0] + y_err[0] + 2.3 * x_pred
y_min_pred = y_avg[0] - y_err[0] + 1.2 * x_pred

# Just so you don't have to guess at the colors...
barcolor, linecolor, fillcolor = 'wheat', 'salmon', 'lightblue'

# Now you're on your own!

# EXERCISE 2.1
# Create the subplots.  For this exercise we are going to overlap the data series

fig, ax = plt.subplots()

#this produces the histograms.  I think the color looks like 'darkgrey' in the soution.  
#I have to align at 'edge' to get the bars to print like the solution
vert_bars=ax.bar(x_pos,y_avg,yerr=y_err,width=bar_width,capsize=2,align='edge',
                 ecolor='darkgrey',edgecolor='darkgrey',color=barcolor)

# for the predicted values, filled inbetween
ax.fill_between(x_pred,y_min_pred,y_max_pred, color=fillcolor)

#for the raw values
ax.plot(x_raw,y_raw,color=linecolor)

#add the titles
ax.set(title='Future Projection of Attitudes', 
       ylabel='Snarkiness (snark units)', 
       xlabel='Minutes since class began')

plt.show()

# EXERCISE 2.2
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)

plt.style.use('classic')

# Generate random data with different ranges...
data1 = np.random.random((10, 10))
data2 = 2 * np.random.random((10, 10))
data3 = 3 * np.random.random((10, 10))

# Set up our figure and axes...
fig, axes = plt.subplots(ncols=3, figsize=plt.figaspect(0.5))
fig.tight_layout() # Make the subplots fill up the figure a bit more...
cax = fig.add_axes([0.25, 0.1, 0.55, 0.03]) # Add an axes for the colorbar

# Now you're on your own!

# EXERCISE 2.2
# Set up our figure and axes...a bit differently than instructed
fig, (ax1, ax2, ax3)=plt.subplots(ncols=3,figsize=plt.figaspect(0.5))

# Make the subplots fill up the figure a bit more...
fig.tight_layout()
# Add an axes for the colorbar
cax = fig.add_axes([0.25, 0.1, 0.55, 0.03]) 

#My solution.  I'm not sure the colors are quite right but everythign else looks good
im=ax1.imshow(data1, interpolation='nearest', cmap='gist_earth',vmin=0, vmax=3)
im=ax2.imshow(data2, interpolation='nearest', cmap='gist_earth',vmin=0, vmax=3)
im=ax3.imshow(data3, interpolation='nearest', cmap='gist_earth',vmin=0, vmax=3)
fig.colorbar(im, orientation='horizontal', cax=cax)
plt.show()