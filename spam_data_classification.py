# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 20:37:10 2021

@author: shubhgoswami
"""

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from joblib import dump

def preprocessor(text):
    """
    Process the text sentence passed as argument.
    HTML tags along with special characters are removed.
    The final sequence is converted into lower case and returned.
    """
    # Remove HTML markup tags.
    text = re.sub('<[^>]*>', '', text)
    
    # Find all valid special characters which can be used to split the text.
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    
    # Convert all characters to lower case and join based on emoticons.
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    
    return text


def read_csv(path):
    """
    path: string. 
        Complete path or relative path to the data csv file.
    """
    return pd.read_csv(path)


def split_train_test(data,x_col,y_col,test_size):
    """
    data: pandas dataframe.
        Dataframe containing data.
    x_col: string or list.
        Column(s) in data that contribute to input features.
    y_col: string
        Column in data containing the output label.
    """
    X = data[x_col]
    y = data[y_col]

    return train_test_split(X, y, test_size = test_size, random_state=42)


def classifier_pipeline():
    """
    Pipeline containing TFIDF Vectorizer to convert text features into numeric 
    features, and then an MLP as a classifier.
    The vocabulary size is being limited to max 800.
    So the first layer in MLP also have 800 neurons.
    """
    tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, 
                            max_features=800, preprocessor=preprocessor, 
                            ngram_range=(1,1))
    clf_pipeline = Pipeline([('vectorizer', tfidf), 
                                    ('nn', MLPClassifier(hidden_layer_sizes=(800, 1000)))])
    
    return clf_pipeline
    
    
def train_pipeline(clf_pipeline, X_train, y_train, model_name):
    """
    Train the pipeline on the data passed.
    Once trained, save the pipeline into file.
    """
    clf_pipeline.fit(X_train, y_train)
    
    dump(clf_pipeline, '{}.joblib'.format(model_name))
    
    return clf_pipeline


def evaluate_pipeline(clf_pipeline,X_test,y_test):
    """
    Using test data, evaluate the trained pipeline.
    """
    y_pred = clf_pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))
    print('Accuracy: {0:.2f} %'.format(100 * round(accuracy_score(y_test, y_pred),4)))
    
    
def main():
    # Read Data.
    data = read_csv("./data/spam_data.csv")
     
    # Split data into train and test.
    X_train, X_test, y_train, y_test = split_train_test(data,"Message",
                                                        "Category",0.3)
    
    # Create classifier pipeline.
    clf_pipeline = classifier_pipeline()
    
    # Train pipeline and save it.
    clf_pipeline = train_pipeline(clf_pipeline, X_train, y_train, 
                                  "spam_classifier")
    
    # Evaluate the trained pipeline.
    evaluate_pipeline(clf_pipeline,X_test,y_test)
                     
    
if __name__=='__main__':
    main()


