import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.metrics import jaccard_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pickle
#import sklearn.external.joblib as extjoblib
import joblib
#from sklearn.externals import joblib

def load_data(database_filepath):
    '''
    Function to load data from the database 
    
    Arguments:
        database_filepath -> Path to SQLite database that was outputed from process_data.py
    
    Outputs:
        X -> A dataframe containing the features (message)
        Y -> A dataframe containing the labels (36 categories in total)
        category_names -> A list of all the names of categories
    '''
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('messages_categories',engine)

    X = df['message'].values #these are the text values of the messages
    Y = df.iloc[:,4:] #these are the 35 categories
    
    category_names = Y.columns.values
    return X,Y,category_names


def tokenize(text):
    '''
    A function to tokenize and clean the text
    
    Arguments:
        text -> The text that should be tokenized 
    
    Outputs:
        clean_tokens -> A list of tokens extracted from text after case normalizing, removing stop words and lemmatizing  
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in words:
        if tok not in stopwords.words("english"):
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)
    return clean_tokens


def build_model_random_forrest():
    '''
    Builds a pipeline using a random forrest as a classifier
    
    Outputs:
        The best model from cross validation of some paramteres of a pipeline that processes text and classifies it using a random forrest classifier
    '''
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
    ])
    
    parameters = {'tfidf__use_idf':[True, False],
              'clf__estimator__criterion': ['gini','entropy'],
              'clf__estimator__n_estimators': [5, 10, 25]}

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose = 2, scoring='f1_micro')
    
    return cv
   
def build_model_logreg():
    '''
    Builds a pipeline using a logistic regression as a classifier
    
    Outputs:
        The best model from cross validation of some paramteres of a pipeline that processes text and classifies it using a logistic regression classifier
    '''
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(LogisticRegression(random_state=42)))
    ])
    
    parameters = {'tfidf__use_idf':[True, False],
              'clf__estimator__C': [0.001,1,10,100]
              }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose = 2, scoring='f1_micro')
    
    return cv

def build_model_svm():
    '''
    Builds a pipeline using a logistic regression as a classifier
    
    Outputs:
        The best model from cross validation of some paramteres of a pipeline that processes text and classifies it using a logistic regression classifier
    '''
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(SVC(random_state=42)))
    ])
    
    parameters = {'tfidf__use_idf':[True, False],
                'clf__estimator__C': [0.001,1,10,100]
                }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose = 2, scoring='f1_micro')
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    To evaluate the performance of the model on test data
    
    Arguments:
        model -> The model to be evaluated
        X_test -> The test data that we want to test the model with
        Y_test -> The labels for the test data
        category_names -> The names of the categories in the test data
        
    Outputs:
        prints the classification report for all the categories as well as the average scores across all categories
    '''
    # predict on test data
    y_pred = model.predict(X_test)

    #print the classsification report for all categories
    print(classification_report(Y_test.values, y_pred, target_names= Y_test.columns.values))
    
    result = precision_recall_fscore_support(Y_test.values, y_pred, average='micro')
    # print the average micro scores
    print(result)
    
    return result[2] #return F1 score
    

def save_model(model, model_filepath):
    '''
    To save a model to a specified filepath
    Arguments:
        model -> The model to be saved
        model_filepath -> The filepath that the model should be saved to
    '''
    pickle.dump(model, open(model_filepath, 'wb'))
    


def main():
    if len(sys.argv) == 3:
        
        database_filepath, model_filepath = sys.argv[1:]
        
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        
        models_to_test = ['RF','LR','SVM']
    
        train_vs_load = 'T'
        best_models = []
        scores = []
        
        for model_name in models_to_test:
            
            if(model_name == "RF"):
                if(train_vs_load == 'T'):
                    print('Building model with a random Forrest classifier...')
                    model = build_model_random_forrest()

                    print('Training model and cross validating the model...')
                    model.fit(X_train, Y_train)
                    
                    best_models.append(model.best_estimator_)
                    
                else:
                    print('Loading pretrained model with random forrest...')
                    model = joblib.load("../models/classifier_rf.pkl")
                    
                    best_models.append(model)
                
                print('Evaluating model...')
                best_f1 = evaluate_model(model, X_test, Y_test, category_names)
                scores.append(best_f1)
            
            elif(model_name == "LR"):
                
                if(train_vs_load == 'T'):
                    print('Building model with a logistic regression classifier...')
                    model = build_model_logreg()
                    
                    print('Training model and cross validating the model...')
                    model.fit(X_train, Y_train)
                    
                    best_models.append(model.best_estimator_)
                    
                else:
                    print('Loading pretrained model with logistic regression...')
                    model = joblib.load("../models/classifier_lr.pkl")
                    
                    best_models.append(model)
                    
                    
                print('Evaluating model...')
                best_f1 = evaluate_model(model, X_test, Y_test, category_names)
                scores.append(best_f1)

            
            elif(model_name == "SVM"):
                
                if(train_vs_load == 'T'):
                    print('Building model with a SVM classifier...')
                    model = build_model_svm()

                    print('Training model and cross validating the model...')
                    model.fit(X_train, Y_train)
                    
                    best_models.append(model.best_estimator_)
                    
                else:
                    print('Loading pretrained model with SVM...')
                    model = joblib.load("../models/classifier_svm.pkl")
                    
                    best_models.append(model)
                    
                    
                print('Evaluating model...')
                best_f1 = evaluate_model(model, X_test, Y_test, category_names)
                scores.append(best_f1)


        print(scores)
        max_score = max(scores)
        ind_max = scores.index(max_score)
        best_model = best_models[ind_max]
        
        print('Saving the best model...\n    MODEL: {}'.format(model_filepath))
        save_model(best_model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')
    

if __name__ == '__main__':
    main()