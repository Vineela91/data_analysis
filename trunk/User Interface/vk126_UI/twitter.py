import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import re

import string
import nltk
nltk.download('stopwords')
from nltk.sentiment.vader import SentimentIntensityAnalyzer as VS
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords
from nltk.stem.porter import *

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import seaborn as sns

plt.style.use('ggplot')

import warnings
warnings.filterwarnings("ignore")
#import nltk
#nltk.download(['stopwords'])#(['punkt','stopwords'])
from nltk.corpus import stopwords
#stopwords = stopwords.words('english')

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from plotnine import ggplot, aes, geom_line,geom_density,geom_point
from sklearn.feature_extraction.text import CountVectorizer
# import these modules
nltk.download('omw-1.4')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer  
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix, classification_report,f1_score,accuracy_score

from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.svm import LinearSVC
# import numpy as np
from sklearn.model_selection import KFold

def sentiment_class(sent_score):
    if sent_score >= 0.05:
        sent_class = 2
    elif -0.05 < sent_score < 0.05:
        sent_class = 1
    else:
        sent_class = 0
    return sent_class

def pre_process(tweet):
## removing spaces
    remove_space = re.compile(r'\s+')
    tweet_1 = tweet.replace(remove_space, ' ')
## removing url
    remove_url =  re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    tweet_2 = tweet_1.replace(remove_url, '')
## removing names
    remove_name = re.compile(r'@[\w\-]+')
    tweet_3 = tweet_2.replace(remove_name, '')
## removing special characters
    tweet_4=tweet_3.replace("[^a-zA-Z]", " ")
    tweet_5=tweet_4.replace(r'\s+', ' ')
    tweet_6=tweet_5.replace(r'^\s+|\s+?$','')
    tweet_7=tweet_6.replace(r'\d+(\.\d+)?','numbr')
    print (tweet_7)
    tweet_8=tweet_7.str.lower()

    return tweet_8
 
def lemmatize(twt):
  Lemmatizer = WordNetLemmatizer()
  lemmatized = ' '.join([Lemmatizer.lemmatize(w) for w in twt])
  return lemmatized

def twitter_process(df):

    df.drop(['Unnamed: 0','Date_time','Tweet_id','User_name','Year','Month'], axis=1, inplace=True)
    df.dropna(inplace=True)
    
    # Generating sentiment score
    stopwords1 = stopwords.words('english')
    analyzer = SentimentIntensityAnalyzer()
    df['tweet_cleaned'] = df['Tweet_text'].apply(lambda x: ' '.join([item for item in x.split() if item not in stopwords1]))
    df['Sentiment_score'] = df['tweet_cleaned'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    df['Sentiment_class'] = df['Sentiment_score'].apply(sentiment_class)
    df['Tweet_len'] = df['Tweet_text'].apply(len)
    df['clean_tweet_len'] = df['tweet_cleaned'].apply(len)
    
    # Plotting
    sns.set_style("darkgrid")
    sns.countplot(x='Sentiment_class', data=df)
    plt.savefig('static/images/imgg1.png')
    plt.clf()
    
    p = ggplot(aes(x='Sentiment_score', y='Sentiment_class'), df) + geom_point()
    p.save('static/images/imgg2.png')
    plt.clf()
    
    cnt_v_1 = CountVectorizer().fit(df['Tweet_text'])
    cnt_v_2 = cnt_v_1.transform(df['Tweet_text'])
    cnt_v_3 = cnt_v_2.sum(axis=0)
    freq_cnt = [(word, cnt_v_3[0, iz]) for word, iz in cnt_v_1.vocabulary_.items()]
    freq_cmt = sorted(freq_cnt, key=lambda x: x[1], reverse=True)
    freq_cmt = freq_cmt[:20]
    word_df = pd.DataFrame(freq_cmt, columns=['Word', 'count'])
    sns.barplot(x='Word', y='count', data=word_df)
    plt.savefig('static/images/imgg3.png')
    plt.clf()
    
    sns.histplot(data=df, x='Sentiment_score')
    plt.savefig('static/images/imgg4.png')
    plt.clf()
    
    sns.displot(data=df, x='Sentiment_score', kind='kde')
    plt.savefig('static/images/imgg5.png')
    plt.clf()
    
    fig_class = sns.FacetGrid(data=df, col='Sentiment_class')
    fig_class.map(plt.hist, 'Tweet_len', bins=50)
    plt.savefig('static/images/imgg6.png')
    plt.clf()
    
    fig_class = sns.FacetGrid(data=df, col='Sentiment_class')
    fig_class.map(plt.hist, 'clean_tweet_len', bins=50)
    plt.savefig('static/images/imgg7.png')
    plt.clf()
    
    # Character Length of Titles - Min, Mean, Max
    print('Mean Length', df['Tweet_text'].apply(len).mean())
    print('Min Length', df['Tweet_text'].apply(len).min())
    print('Max Length', df['Tweet_text'].apply(len).max())
    
    x = df['Tweet_text'].apply(len).plot.hist()
    
    plt.savefig("static/images/imgg8.png", dpi=300)
    
    """# preprocessing tweets"""



    processed_tweets_1= pre_process(df['Tweet_text']) # preprocessing the tweets
    
    print(processed_tweets_1)
    
    
    Lemmatizer = WordNetLemmatizer()
    
    
    """* tokenizing the tweets"""
    
    twt_tokenize = processed_tweets_1.apply(lambda x: x.split()) # tokenizing the tweets
    print(twt_tokenize)
    
    """* removing the stop words"""
    
    twt_stopwords=twt_tokenize.apply(lambda x: [item for item in x if item not in stopwords1])
    print(twt_stopwords)
    
    """* lemmatizing the tweets"""



    twt_lemma = twt_stopwords.apply(lemmatize)
    print(twt_lemma)
    
    df['processed_tweets']= twt_lemma
    df.head()
    
    df.reset_index(drop=True,inplace=True)
    
    # df.loc[17992]=df.loc[17993].copy()
    df.loc[17991:17995]
    
    cnt=0
    # while cnt <20:
    # while cnt <20:
    cnt_v_1=CountVectorizer().fit(df['Tweet_text']) 
    cnt_v_2=cnt_v_1.transform(df['Tweet_text'])
    cnt_v_3=cnt_v_2.sum(axis=0)
    freq_cnt = [(word, cnt_v_3[0, iz]) for word, iz in cnt_v_1.vocabulary_.items()]
    freq_cmt =sorted(freq_cnt, key = lambda x: x[1], reverse=True)
    freq_cmt=freq_cmt[:20]
    word_df = pd.DataFrame(freq_cmt, columns = ['Word' , 'count'])
    
    plt.figure(figsize = (20, 5))
    sns.barplot(x ='Word',y='count', data = word_df)
    plt.savefig("static/images/imgg9.png", dpi=300)
    
    """* count of sentiment score"""
    
    sns.histplot(data=df, x="Sentiment_score")#,kind='kde')#,hue='Sentiment_class',kind="ecdf")
    plt.savefig("static/images/imgg10.png", dpi=300)
    """* distribution of sentiment score"""
    
    sns.displot(data=df, x="Sentiment_score",kind='kde')#,hue='Sentiment_class',kind="ecdf")
    
    """* comparison of tweet lengths for different sentiment classes"""
    
    plt.figure(figsize=(20,10))
    fig_class = sns.FacetGrid(data=df, col='Sentiment_class')
    fig_class.map(plt.hist, 'Tweet_len', bins=50)
    plt.savefig("static/images/imgg11.png", dpi=300)
    plt.figure(figsize=(20,10))
    fig_class = sns.FacetGrid(data=df, col='Sentiment_class')
    fig_class.map(plt.hist, 'clean_tweet_len', bins=50)
    plt.savefig("static/images/imgg12.png", dpi=300)
    # Character Length of Titles - Min, Mean, Max
    print('Mean Length', df['Tweet_text'].apply(len).mean())
    print('Min Length', df['Tweet_text'].apply(len).min())
    print('Max Length', df['Tweet_text'].apply(len).max())
    
    x = df['Tweet_text'].apply(len).plot.hist()
    
    """# preprocessing tweets"""
    
    
    
    processed_tweets_1= pre_process(df['Tweet_text']) # preprocessing the tweets
    
    print(processed_tweets_1)
    
    
    
    
    # def more_process(tweet):
    #   ## tokenizing the tweets
    #   twt1 = tweet.apply(lambda x: x.split())
    
    #   ## remove stop_words
    #   twt2=twt1.apply(lambda x: [item for item in x if item not in stopwords])
    
    #   ## lemmatizing the tweets
    #   twt3 = ' '.join([Lemmatizer.lemmatize(w) for w in twt2])
    
    #   # twt4=twt_stopword.apply(lemma1)
    #   return(twt3)
    
    """* tokenizing the tweets"""
    
    twt_tokenize = processed_tweets_1.apply(lambda x: x.split()) # tokenizing the tweets
    print(twt_tokenize)
    
    """* removing the stop words"""
    
    twt_stopwords=twt_tokenize.apply(lambda x: [item for item in x if item not in stopwords1])
    print(twt_stopwords)
    
    """* lemmatizing the tweets"""
    
   
    
    twt_lemma = twt_stopwords.apply(lemmatize)
    print(twt_lemma)
    
    df['processed_tweets']= twt_lemma
    df.head()
    
    df.reset_index(drop=True,inplace=True)
    
    # df.loc[17992]=df.loc[17993].copy()
    df.loc[17991:17995]
    
    cnt=0
    # while cnt <20:
    cnt_v_1=CountVectorizer().fit(df['processed_tweets']) 
    cnt_v_2=cnt_v_1.transform(df['processed_tweets'])
    cnt_v_3=cnt_v_2.sum(axis=0)
    freq_cnt = [(word, cnt_v_3[0, idx]) for word, idx in cnt_v_1.vocabulary_.items()]
    freq_cmt =sorted(freq_cnt, key = lambda x: x[1], reverse=True)
    freq_cmt=freq_cmt[:20]
    word_df = pd.DataFrame(freq_cmt, columns = ['Word' , 'count'])
    
    # print(freq_cmt[0][1])
    
    """* finding the top 20 words with highest count (without stopwords)"""
    
    plt.figure(figsize = (20, 5))
    sns.barplot(x ='Word',y='count', data = word_df)
    plt.savefig("static/images/imgg13.png", dpi=300)
    """plotting word clouds"""
    
    
    negative_twts = ' '.join([text for text in df['processed_tweets'][df['Sentiment_class'] == 0]])
    wordcloud = WordCloud(width=900, height=500,random_state=34, max_font_size=110).generate(negative_twts)
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    
    plt.savefig("static/images/imgg14.png", dpi=300)
    plt.show()
    neutral_twts = ' '.join([text for text in df['processed_tweets'][df['Sentiment_class'] == 1]])
    wordcloud = WordCloud(width=900, height=500,random_state=34, max_font_size=110).generate(neutral_twts)
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    
    plt.savefig("static/images/imgg15.png", dpi=300)
    plt.show()
    positive_twts = ' '.join([text for text in df['processed_tweets'][df['Sentiment_class'] == 2]])
    wordcloud = WordCloud(width=900, height=500,random_state=55, max_font_size=110).generate(positive_twts)
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    
    plt.savefig("static/images/imgg16.png", dpi=300)
    plt.show()
    """TFID vectorizing tweets"""
    
    
    #determining TF-IDF Features of the words in the dataset. 
    TfId_vectorizer = TfidfVectorizer(ngram_range=(1, 2),max_df=0.75, min_df=5, max_features=10000)
    TfId_vector = TfId_vectorizer.fit_transform(df['processed_tweets'] )
    
    """# Models"""
    
    X=TfId_vector #Tf_vector
    y = df['Sentiment_class'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    
    """### Logistic regression"""
    
    model = LogisticRegression(multi_class='multinomial')
    # model = LogisticRegression(random_state = 42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    acc_1=accuracy* 100
    print(f"Accuracy of Logistic Regression {acc_1:.2f} %")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print('  ')
    print(report)
    
    """## Random Forest classifier"""
    
    model = RandomForestClassifier(random_state = 42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    acc_1=accuracy* 100
    print(f"Accuracy of Logistic Regression {acc_1:.2f} %")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print('  ')
    print(report)
    
    """## Linear support vector classifier"""
    
    
    model = LinearSVC()
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    acc5=acc* 100
    print(f"Accuracy of Support vector classifier {acc5:.2f} %")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print('  ')
    print(report)
    
    """## Multinomial naive bayes"""
    
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    acc4=acc* 100
    print(f"Accuracy of Multinomial Naive Bayes {acc4:.2f} %")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print('  ')
    print(report)
    
    y_train.value_counts() # displaying value counts of target variable
    
    """* eliminating class imbalance using smote"""
    
    
    # Osmple = SMOTE()
    Osmple = SMOTE(random_state=50)
    X1, y1 = Osmple.fit_resample(X_train, y_train)
    
    # summarize distribution
    counter = Counter(y1)
    for k,v in counter.items():
    	per = v / len(y1) * 100
    	print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
    
    """## Logistic regression with upsampling"""
    
    model = LogisticRegression(random_state = 42)
    model.fit(X1, y1)
    accuracy = model.score(X_test, y_test)
    acc_1=accuracy* 100
    print(f"Accuracy of Logistic Regression {acc_1:.2f} %")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print('  ')
    print(report)
    
    """## Random Forest classifier with upsampling"""
    
    model = RandomForestClassifier(random_state = 42)
    model.fit(X1, y1)
    accuracy = model.score(X_test, y_test)
    acc_1=accuracy* 100
    print(f"Accuracy of Logistic Regression {acc_1:.2f} %")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print('  ')
    print(report)
    
    """## Linear support vector classifier with upsampling"""
    
    
    model = LinearSVC()
    model.fit(X1, y1)
    acc = model.score(X_test, y_test)
    acc5=acc* 100
    print(f"Accuracy of Support vector classifier {acc5:.2f} %")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print('  ')
    print(report)
    
    """## Multinomial naive bayes with upsampling"""
       
    
    model = MultinomialNB()
    model.fit(X1, y1)
    acc = model.score(X_test, y_test)
    acc4=acc* 100
    print(f"Accuracy of Multinomial Naive Bayes {acc4:.2f} %")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print('  ')
    print(report)
    
    """# Using K-Fold Validation"""
    
    kf = KFold(n_splits=10)
    accu=[]
    i=1
    for train_index, test_index in kf.split(X):
      # print(train_index)
      # print(test_index)
        # print('-------------------------------------')
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = LogisticRegression(multi_class='multinomial')
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        acc_1=accuracy* 100
        print(f"Accuracy of Logistic Regression model {i} {acc_1:.2f} %")
        accu.append(acc_1)
        i=i+1
    
    print(' ')
    print(f"The average accuracy of the 10 models for logistic regression is {sum(accu)/10:.2f} %")
    
    kf = KFold(n_splits=10)
    accu=[]
    i=1
    for train_index, test_index in kf.split(X):
      # print(train_index)
      # print(test_index)
        # print('-------------------------------------')
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = RandomForestClassifier()#(multi_class='multinomial')
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        acc_1=accuracy* 100
        print(f"Accuracy of Random Forest Classifier model {i} {acc_1:.2f} %")
        # y_pred = model.predict(X_test)
        # report = classification_report(y_test, y_pred)
        # print('  ')
        # print(report)
        # print(' ')
        accu.append(acc_1)
        i=i+1
    
    print(' ')
    print(f"The average accuracy of the 10 models for Random Forest Classifier is {sum(accu)/10:.2f} %")
    
    kf = KFold(n_splits=10)
    accu=[]
    i=1
    for train_index, test_index in kf.split(X):
      # print(train_index)
      # print(test_index)
        # print('-------------------------------------')
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = LinearSVC()#(multi_class='multinomial')
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        acc_1=accuracy* 100
        print(f"Accuracy of Linear SVC model {i} {acc_1:.2f} %")
        # y_pred = model.predict(X_test)
        # report = classification_report(y_test, y_pred)
        # print('  ')
        # print(report)
        # print(' ')
        accu.append(acc_1)
        i=i+1
    
    print(' ')
    print(f"The average accuracy of the 10 models for Linear SVC is {sum(accu)/10:.2f} %")
    
       
    kf = KFold(n_splits=10)
    accu=[]
    i=1
    for train_index, test_index in kf.split(X):
      # print(train_index)
      # print(test_index)
        # print('-------------------------------------')
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = MultinomialNB()#(multi_class='multinomial')
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        acc_1=accuracy* 100
        print(f"Accuracy of Multinomial Naive bayes {i} {acc_1:.2f} %")
        # y_pred = model.predict(X_test)
        # report = classification_report(y_test, y_pred)
        # print('  ')
        # print(report)
        # print(' ')
        accu.append(acc_1)
        i=i+1
    
    print(' ')
    print(f"The average accuracy of the 10 models for Multinomial Naive bayes is {sum(accu)/10:.2f} %")
