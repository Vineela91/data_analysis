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
import sys 
warnings.filterwarnings("ignore")



def covid(df):
    df_1=df
    plt.figure(figsize=(8,5))
    plt.plot(df_1['date'], df_1['7 day rolling death count'])
    plt.title("7 day rolling death count")
    # Save the figure as a PNG file
    plt.savefig('static/images/rolling_death_count1.png')
    
    # Show the figure
    plt.show()
    y = df_1['7 day rolling death count']
    x = df_1['date']
    
    plot = pxg.Figure(data=[pxg.Scatter(x=x, y=y, mode='markers')])
    
    plot.update_layout(
        title="Rolling Death Count",
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, step="day", stepmode="backward")
                ])
            ),
            rangeslider=dict(visible=True)
        )
    )
    
    # Save the figure as a PNG file
    pio.write_image(plot, 'static/images/rolling_death_count2.png')
    
    # Show the figure
    plot.show()
    fig = px.histogram(df_1, x="date", y='7 day rolling death count')
    
    # Save the figure as a PNG file
    pio.write_image(fig, 'static/images/rolling_death_count_hist.png')
    
    # Show the figure
    fig.show()
    plot = pxg.Figure(data=[pxg.Scatter(
    	x=df_1['7 day rolling death count'],
    	y=df_1['date'],
    	mode='markers')
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
    
    # Set title
    plot.update_layout(title='Rolling Death Count Scatter Plot')
    
    # Save the figure as a PNG file
    pio.write_image(plot, 'static/images/rolling_death_count_scatter.png')
    
    # Show the figure
    plot.show()