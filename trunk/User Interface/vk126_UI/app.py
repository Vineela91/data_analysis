from flask import Flask, render_template, redirect, url_for, request, flash, session
import csv
import pandas as pd 
import plotly.express as px
import plotly.graph_objects as pxg
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.io as pio
import datetime
import warnings
import sys 
warnings.filterwarnings("ignore")

from plotnine import ggplot, aes, geom_line,geom_density,geom_point
from twitter import twitter_process
from claimage import claimantfunction
from covidmorality import covid

app = Flask(__name__)
app.secret_key = 'mysecretkey'  # Change this to your secret key

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/filterpage', methods=['GET', 'POST'])
def filterpage():
    month1=0
    year=0
    search_word=''#'election'#'leicester'--- give '' for no search word
    if request.method == 'POST':
            # Retrieve the user's cre
            tweet_count = int(request.form['tweetcount'])
            month1 = request.form['month']
            year = request.form['year']
            search_word = request.form['keyword']
            
            if tweet_count == 5000:
            # Data reading
                ndf=pd.read_csv('leicester_city_twitter_data_5k.csv',parse_dates=['Date_time'])
            elif tweet_count == 10000:
            # Data reading
                ndf=pd.read_csv('leicester_city_twitter_data_10k.csv',parse_dates=['Date_time'])
            elif tweet_count == 20000:
            # Data reading
                ndf=pd.read_csv('leicester_city_twitter_data_20k.csv',parse_dates=['Date_time'])
                ndf['Date_time'] = pd.to_datetime(ndf['Date_time'])
            else:
                flash('You must enter tweet counts as 5000, 10000 or 20000!')
            # Create separate columns for month and year
            ndf['Month'] = ndf['Date_time'].dt.month
            ndf['Year'] = ndf['Date_time'].dt.year
            
           
            
            if month1.lower()=='all':
              X1=ndf.copy()
            else:
              month = datetime.datetime.strptime(month1, '%B').month
              X1=ndf[ndf['Month']==int(month)]
            if year.lower()=='all':
              X2=X1.copy()
            else:
              X2=X1[X1['Year']==int(year)]
            
            print(X2.columns)
            if search_word.lower()=='all':
              df=X2.copy()
            else:
              # df=X2.loc[X2['Text'].isin(search_word)]
              try:
                  df= X2[X2["Tweet_text"].str.contains(f'{search_word}')]
                  if df.shape[0] == 0:
                    print('No tweets found')
                    
                    sys.exit()
              except:
                  flash('No tweets found!')
                  return render_template('twitter_analysis5.html')
            twitter_process(df)

            return render_template('twitter_analysis5.html')
        
    return render_template('filterpage.html')



@app.route('/filterpageage', methods=['GET', 'POST'])
def filterpageage():
    
    if request.method == 'POST':
        try:

            year = request.form['year']
            keyword = request.form['keyword']
            location = request.form['location']
        
            ndf=pd.read_csv('claimant-count-by-age-and-sex.csv',parse_dates=['DATE'],sep=";")
            # Create separate columns for month and year
            ndf['Month'] = ndf['DATE'].dt.month
            ndf['Year'] = ndf['DATE'].dt.year
            
            
            if year.lower()=='all':
                mdf=ndf.copy()
            else:
                mdf=ndf[ndf['Year']==int(year)]
            if keyword.lower()=='all':
                ldf=mdf.copy()
            else:
                month = datetime.datetime.strptime(keyword, '%B').month
                ldf=mdf[mdf['Month']==int(month)]
            if location.lower()=='all':
                df=ldf.copy()
            else:    
            
                df=ldf[ldf['Location']==location]
            print(df)
            
            claimantfunction(df)
            return render_template('claimant_count_age_sex.html')
        except:
            flash('Data does not exist!')
           
    return render_template('filterpageage.html')


@app.route('/filterpageage1', methods=['GET', 'POST'])
def filterpageage1():
    
        if request.method == 'POST':
            try:
                # Retrieve the user's cre
                year = request.form['year']
                keyword = request.form['keyword']
                
            
                ndf=pd.read_excel("covid-19-mortality-within-28-days-of-diagnosis.xlsx")
                ndf['date'] = pd.to_datetime(ndf['date'])
                # Create separate columns for month and year
                ndf['Month'] = ndf['date'].dt.month
                ndf['Year'] = ndf['date'].dt.year
                
                
                if year.lower()=='all':
                    mdf=ndf.copy()
                else:
                    mdf=ndf[ndf['Year']==int(year)]
                if keyword.lower()=='all':
                    ldf=mdf.copy()
                else:
                    month = datetime.datetime.strptime(keyword, '%B').month
                    ldf=mdf[mdf['Month']==int(month)]
                
                covid(ldf)
                return render_template('covid_morality.html')
            except:
                flash('Data does not exist!')
            
        return render_template('filterpageage1.html')
    


@app.route('/filterpageage2', methods=['GET', 'POST'])
def filterpageage2():
    if request.method == 'POST':
        try:
                # Retrieve the user's cre
                year = request.form['year']
                
                          
                
                df_1=pd.read_csv("strategic-cordon-data-totals.csv",delimiter=';')
                
                if year !='all':
                    df_2009 = df_1[df_1["Year"] == int(year)]
                
                
                fig = px.scatter(df_2009, x="Vehicle Type", y="value", color='Cordon')
                
                # Set title
                fig.update_layout(title='Vehicle Type Scatter Plot')
                
                # Save the figure as a PNG file
                pio.write_image(fig, 'static/images/vehicle_type_scatter1.png')
                
                # Show the figure
                fig.show()
                
                fig = px.histogram(df_2009, x="value", y="Cordon", color='Vehicle Type')
                
                # Set title
                fig.update_layout(title='Vehicle Type Histogram')
                
                # Save the figure as a PNG file
                pio.write_image(fig, 'static/images/vehicle_type_histogram.png')
                
                # Show the figure
                fig.show()
                
                x = df_2009['value']
                y = df_2009['Vehicle Type']
                
                fig = go.Figure(data=[go.Scatter(
                    x=x,
                    y=y,
                    mode='markers',
                )])
                
                fig.update_layout(
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
                    ),
                    title='Vehicle Type Scatter Plot'
                )
                
                # Save the figure as a PNG file
                pio.write_image(fig, 'static/images/vehicle_type_scatter3.png')
                
                # Show the figure
                fig.show()
                
                plt.hist(df_2009['Vehicle Type'])
                plt.title('Vehicle Type Distribution')
                plt.xlabel("Vehicle Type")
                plt.ylabel("Frequency")
                plt.savefig('static/images/vehicle_type_hist2009.png')
                plt.show()
                return render_template('Strategic_cordon_survey.html')
        except:
            flash('Data does not exist!')
                
    return render_template('filterpageage2.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Retrieve the user's credentials from the login form
        username = request.form['username']
        password = request.form['password']
        
        # Check if the user's credentials are valid
        with open('users.csv', 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['username'] == username and row['password'] == password:
                    # Set a session variable to remember the user's username
                    session['username'] = username
                    # Redirect the user to the welcome page
                    return redirect(url_for('welcome'))
        
        # If the user's credentials are invalid, show an error message
        flash('Invalid username or password. Please try again.', 'danger')
        
    return render_template('login.html')

@app.route('/forgot', methods=['GET', 'POST'])
def forgot():
    if request.method == 'POST':
        # Retrieve the user's credentials from the login form
        username = request.form['username']
        Question = request.form['Question']
        password = request.form['password']
        
       
        
        
        # Check if the user's credentials are valid
        with open('users.csv', 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['username'] == username and row['Question'] == Question:
                    df = pd.read_csv('users.csv')
                    
                    # Define the username to match
                    username_to_delete = username
                    
                    
                    # Find the index of the row(s) where the username column matches the specified value
                    index_to_delete = df[df['username'] == username_to_delete].index
                    
                    # Drop the row(s) with the matching username
                    df = df.drop(index_to_delete)

                    # Rewrite the updated data back to the CSV file
                    df.to_csv('users.csv', index=False)
                    with open('users.csv', 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([username, password,Question])
                    return redirect(url_for('login'))
        
        # If the user's credentials are invalid, show an error message
        flash('Wrong answer. Please try again.')
        
    return render_template('forgot.html')





@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Retrieve the user's credentials from the signup form
        username = request.form['username']
        password = request.form['password']
        Question = request.form['Question']
        
        # Check if the user's credentials are valid
        with open('users.csv', 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['username'] == username:
                    flash('Username already exists. Please choose a different username.', 'danger')
                    return redirect(url_for('signup'))
        
        # If the username does not already exist, add the user's credentials to the CSV file
        with open('users.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([username, password,Question])
        
        # Set a session variable to remember the user's username
        session['username'] = username
        
        # Redirect the user to the welcome page
        return redirect(url_for('welcome'))
        
    return render_template('signup.html')

@app.route('/welcome')
def welcome():
    # Retrieve the user's username from the session variable
    username = session.get('username', None)
   
    if username:
        return render_template('welcome.html', username=username)
    else:
        return redirect(url_for('login'))


if __name__ == '__main__':
    app.run()
