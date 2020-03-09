import sys

import warnings
warnings.filterwarnings('ignore')

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])

import pandas as pd
import numpy as np
import sqlalchemy
import sqlite3
from sqlalchemy import create_engine
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression


def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))

    df = pd.read_sql_table("DisasterResponse", con=engine)

    #df = df.ix[:899,]

    y_columns = ['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']
    X = df['message']
    y = df[y_columns]

    return X, y, y_columns


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-z0-9]"," ", text)

    tokens = word_tokenize(text)

    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    tokens = [lemmatizer.lemmatize(word).strip() for word in tokens if word not in stop_words]

    return tokens



def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])

    parameters = {
        #'vect__ngram_range':((1,1),(1,2)),
        #'vect__max_df':(0.5,0.75,1.0),
        #'vect__max_features':(None, 5000, 10000),
        #'tfidf__use_idf':(True,False),
        'clf__estimator__n_estimators': [1, 2, 3]
        #clf__estimator__C" : [0.1,1,10]
        }

    model = GridSearchCV(pipeline, param_grid=parameters)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    y_test_np = Y_test.to_numpy()

    #print(classification_report(Y_test, y_pred, target_names = category_names))

    for i, label in enumerate(category_names):
        print(label)
        print(classification_report(list(y_test_np[:,i]), list(y_pred[:,i])))

    #print("\nBest Parameters:", model.best_params_)


def save_model(model, model_filepath):

    pickle.dump(model, open(model_filepath, 'wb'))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print("\nBest Parameters:", model.best_params_,"\n")
        model = model.best_estimator_

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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
