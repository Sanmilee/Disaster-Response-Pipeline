import os
import pandas as pd
import numpy as np
import pickle
import nltk
import re, sys

from sqlalchemy import create_engine
import sqlite3

from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report
from sklearn.metrics import precision_recall_fscore_support
from scipy.stats import hmean
from scipy.stats.mstats import gmean
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

# python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
# python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl


def load_data(database_filepath):
    '''
    input:
        database_filepath: load the database conataining table and attributes
    output:
        X: attributes for message column 
        y: attributes for the last 36 columns
        category_names: names of the last 36 columns 
    '''

    table_name = 'InsertTableName'
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_query('SELECT * FROM messages_disaster', engine)
    
    # load data from database with 'X' as attributes for message column 
    X = df["message"]
    # load data from database with 'Y' attributes for the last 36 columns
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)


    return X, Y

    

def tokenize(text):
    '''
    input:
        text: input text data containing attributes
    output:
       clean_tokens: cleaned text without unwanted texts
    '''
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # take out all punctuation while tokenizing
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    
    # lemmatize as shown in the lesson
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens




def build_model():
    '''
    Description:
        RandomForest Classifier Model for training and tuning the data
    output:
       cv: RandomForest Classifier trained model 
    '''

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])

    parameters = {'clf__estimator__max_depth': [10, 50, None], 
                  'clf__estimator__min_samples_leaf':[2, 5, 10]}

    cv = GridSearchCV(pipeline, parameters)

    return cv



def evaluate_model(model, X_test, y_test):
    '''
    Input:
        model: RandomForest Classifier trained model
        X_test: Test training features
        Y_test: Test training response variable
    Output:
        None: 
            Display model precision, recall, f1-score, support 
    '''
    y_pred = model.predict(X_test)
    for item, col in enumerate(y_test):
        print(col)
        print(classification_report(y_test[col], y_pred[:, item]))
    


def save_model(trained, trained_model_file):
    '''
    Input:
        trained: RandomForest Classifier trained model
        trained_model_file: trained model name to be saved as pickle
    Output:
        None 
    '''
    pickle.dump(trained, open(trained_model_file, 'wb'))


    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train.as_matrix(), Y_train.as_matrix())
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
    
    