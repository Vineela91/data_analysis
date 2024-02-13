import csv
import pandas as pd 
import plotly.express as px
import plotly.graph_objects as pxg
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.io as pio
import datetime
import subprocess


import warnings
warnings.filterwarnings("ignore")

from plotnine import ggplot, aes, geom_line,geom_density,geom_point
from twitter import twitter_process

def claimantfunction(df_1):

    # create the scatter plot
    fig = px.scatter(df_1, x='Claimant Count', y='Claimant Rate (% of population)', title="claimant count vs claimant rate (%)")
    # save the plot as a PNG image
    fig.write_image("static/images/claimant_rate1.png", width=8, height=8)
    
    fig = px.scatter(df_1, x="Age Group", y="Claimant Count", color='Gender', title="Age Group vs claimant count")
    # save the plot as a PNG image
    fig.write_image("static/images/Age_Group1.png", width=8, height=8)
    
    fig = px.histogram(df_1, x="Claimant Count", y="Age Group", color='Location', title="claimant count vs Age Group")
    # save the plot as a PNG image
    fig.write_image("static/images/Age_Group_location1.png", width=8, height=8)
    
    plot = pxg.Figure(data=[pxg.Scatter(
        x=df_1['Claimant Count'],
        y=df_1['Age Group'],
        mode='markers',)
      # color='Location',)
    ])
    
    # Add dropdown
    plot.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=["type", "scatter"],
                        label="Scatter Plot",
                        method="restyle"
                    ),
                    dict(
                        args=["type", "histogram"],
                        label="Histogram",
                        method="restyle"
                    )
                ]),
            ),
        ]
    )
    
    plot.show()
    # Save the figure as a PNG image
    pio.write_image(plot, 'static/images/age_scatter1.png', width=8, height=8)
    
    
    # df_1 = px.df_1.tips()
    y=df_1['Claimant Rate (% of population)']#Claimant Count'],#Claimant Rate (% of population)
    x=df_1['Population']#DATE'],
    
    plot = pxg.Figure(data=[pxg.Scatter(
        x=x,
        y=y,
        mode='markers',)
    ])
    
    plot.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                        step="day",
                        stepmode="backward"),
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
        )
    )
    
    # Save the plot as an image
    pio.write_image(plot, 'static/images/population1.png', width=8, height=8)
    fig = px.histogram(df_1, x="Gender", title="Gender Distribution")
    # save the plot as a PNG image
    fig.write_image("static/images/gender_distribution1.png", width=8, height=8)
    
    
    plt.figure(figsize=(8,8))
    plt.hist(df_1['Age Group'])
    plt.title("Histogram of age group")
    
    # Save the figure as PNG with 300 dpi resolution
    plt.savefig("static/images/histogram1.png")
    plt.close() 
    
   