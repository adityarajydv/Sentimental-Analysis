# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 12:10:50 2021

@author: ADITYA RAJ YADAV
"""


# Importing the libraries
import pickle
import pandas as pd
import webbrowser
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
project_name = "Sentiment Analysis with Insights"

def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050/")
    
def load_model():
    global pickle_model
    global vocab
    global scrappedReviews
    
    
    scrappedReviews = pd.read_csv('scrappedReviews.csv')
    
    file = open("pickle_model.pkl", 'rb') 
    pickle_model = pickle.load(file)

    file = open("feature.pkl", 'rb') 
    vocab = pickle.load(file)
    
def check_review(reviewText):
    
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vocab)  
    vectorised_review = transformer.fit_transform(loaded_vec.fit_transform([reviewText]))
    return pickle_model.predict(vectorised_review)

def load_data():
    global df
    df = pd.read_csv('balanced_reviews.csv')
    df.dropna(inplace = True)
    df = df[df['overall'] != 3]
    df['Positivity'] = np.where(df['overall'] > 3, 1, 0 )
    df['Names'] = np.where(df['Positivity']==1,'Positive','Negative')
    global labels
    labels = df['Names'].tolist()

def load_scrappeddata():
    global df1
    df1 = pd.read_csv('scrappedReviews.csv')
    global reviews
    reviews = []
    for i in range(len(df1)):
        reviews.append({'label':df1['reviews'][i],'value':i})

def predict_scrappeddata():
    global sentiment
    sentiment = []
    for i in range (len(df1['reviews'])):
        response = check_review(df1['reviews'][i])
        if (response[0]==1):
            sentiment.append('Positive')
        elif (response[0] ==0 ):
            sentiment.append('Negative')
        else:
            sentiment.append('Unknown')
            
def create_app_ui():
    main_layout = html.Div(
            [
                html.H1(id = 'Main_title', children = 'Sentiment analysis with insights',
                        style={'text-align':'left'}),
                html.Div(
            [
                html.H2('Pie chart distribution of Amazon reviews training set  '),
                dcc.Graph(
                            id='graph-1',
                            figure={
                                'data':[go.Pie(labels=labels)],
                                'layout':go.Layout(
                                                    title = 'Sentiment Visualization')
                            }
                         ),
            ],
                style = {"border":"1px outset red",'marginBottom':'1.0em','background-color':'lightblue'}         
            ),
                html.Div(
            [
                html.H2('Pie chart distribution of etsy webscrapped data used for predicting'),
                dcc.Graph(
                            id='graph-2',
                            figure={
                                'data':[go.Pie(labels=sentiment)],
                                'layout':go.Layout(
                                                    title = 'Sentiment Visualization')
                            }
                         ),
            ],
                style = {"border":"1px outset red",'marginBottom':'1.0em','background-color':'lightblue'}         
            ),
                html.Div(
            [
                html.H2('Check the sentiment of scrapped reviews here'),
                html.Label('Pick a review from below checkbox'),
                dcc.Dropdown(
                            id = 'reviewpicker',options = reviews, value=None
                            ),
                dbc.Button(
                            id="check_review", children='Submit',
                            color = 'dark',style = {'width': '100%'}
                          ),
                html.H1(id = 'result2',children = None),
            ],
                style = {"border":"1px outset red",'marginBottom':'1.0em','background-color':'lightblue'}
            ),
                html.Div(
            [
                html.H2('Type a review and click submit to see the prediction'),
                dcc.Textarea(
                            id = 'textarea_review',
                            placeholder = 'Enter the review here...',
                            style={'width': '100%', 'height': 100}
                            ),
                dbc.Button(
                            id="button_review", children='Submit',
                            color = 'dark',
                            style = {'width': '100%'}),                            
                html.H1(id = 'result', children = None),
            ],
                style = {"border":"1px outset red",'marginBottom':'1.0em','background-color':'lightblue'}   
            )
            ],
            style={'border':'1px black dotted'})
    return main_layout

@app.callback(
    Output('result', 'children'),  
    [
    Input('button_review', 'n_clicks')
    ]
    ,
    [
    State('textarea_review', 'value')                                    
    ]                                    
    )                                      
def review_predict(n_clicks,textarea_value):         
    print("Data Type  = ", str(type(n_clicks)))  
    print("Value      = ", str(n_clicks))
    
    print("Data Type  = ", str(type(textarea_value)))
    print("Data Type  = ", str(textarea_value))
    response = check_review(textarea_value)
    if (n_clicks > 0):              
        if (response[0] == 0 ):
            result = 'Negative'
        elif (response[0] == 1 ):
            result = 'Positive'
        else:
            result = 'Unknown'
        
        return result
    else:
        return ""
    
@app.callback(
    Output('result', 'style'),  
    [
    Input('button_review', 'n_clicks')
    ]
    ,
    [
    State('textarea_review', 'value')                                    
    ]                                    
    )                                      
def review_predict(n_clicks,textarea_value):         
    print("Data Type  = ", str(type(n_clicks)))  
    print("Value      = ", str(n_clicks))
    
    print("Data Type  = ", str(type(textarea_value)))
    print("Data Type  = ", str(textarea_value))
    response = check_review(textarea_value)
    if (n_clicks > 0):              
        if (response[0] == 0 ):
            result = {'color':'red'}
        elif (response[0] == 1 ):
            result = {'color':'green'}
        else:
            result = 'Unknown'
        
        return result
    else:
        return ""
    
@app.callback(
    Output('result2','children'),
    [
        Input('check_review','n_clicks')
    ],
    [
        State('reviewpicker','value')
    ])
def review_predict2(n_clicks,value):
    review_selected = reviews[value]['label']
    response = check_review(review_selected)
    if (n_clicks>0):
        if (response[0]==0):
            result = 'Negative'
        elif (response[0]==1):
            result = 'Positive'
        else:
            result = 'Unknown'
        return result
    else:
        return ""
    
@app.callback(
    Output('result2','style'),
    [
        Input('check_review','n_clicks')
    ],
    [
        State('reviewpicker','value')
    ])
def review_predict2(n_clicks,value):
    review_selected = reviews[value]['label']
    response = check_review(review_selected)
    if (n_clicks>0):
        if (response[0]==0):
            result = {'color':'red'}
        elif (response[0]==1):
            result = {'color':'green'}
        else:
            result = 'Unknown'
        return result
    else:
        return ""
    
def main():
    global app
    global project_name
    load_model()
    load_data()
    load_scrappeddata()
    predict_scrappeddata()
    open_browser()
    app.title = project_name
    app.layout = create_app_ui()
    app.run_server()
    app = None
    project_name = None
    
    
if __name__ == '__main__':
  main()
