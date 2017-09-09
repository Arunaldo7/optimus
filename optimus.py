# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 14:18:04 2017

@author: arunk
"""

import pickle;
import re;
from nltk.stem.lancaster import LancasterStemmer;
from nltk.corpus import stopwords;
import json;
import random;
from subprocess import call;

#cleaning the texts
def clean_texts(user_query):
    corpus = [];
    #retain only alphabets
    review = re.sub("[^a-z A-Z]", " ", user_query);
    review = review.lower();
    clean_text = review.split();
    ls = LancasterStemmer();
    review = [ls.stem(word) for word in clean_text if word not in set(stopwords.words("english"))];         
    if(len(review) < 2):
        clean_text = [ls.stem(word) for word in clean_text];
    else:
        clean_text = review;
    clean_text = " ".join(clean_text); 
    corpus.append(clean_text)
    return corpus;

#create bag of words model from saved vectorizer
def bow(vectorizer,cleaned_query):
    x_test = vectorizer.transform(cleaned_query).toarray();
    return x_test;

#encode dependent variable and predict the output
def predict(classifier,label_encoder_y,x_test):
    y_pred = classifier.predict(x_test);
    decoder = label_encoder_y.inverse_transform(y_pred);
    return decoder;

#response
def response(intent_file,intent):
    
    with open(intent_file,'r') as json_data:
        dataset = json.load(json_data);
    
    #clean training texts in corpus
    #loop through each sentence in our intents patterns
    for train_data in dataset['train_set']:
            if train_data['intent'] == intent:
                    return random.choice(train_data['responses']);
                
#process and predict intent for user query
def process_query(user_query):
    
    #clean user input query
    cleaned_query = clean_texts(user_query);
    print(cleaned_query)
    #load pickle data for classifier and encoder
    trained_data = pickle.load(open('training_data.pkl','rb'));
    
    classifier = trained_data['classifier'];
    vectorizer = trained_data['vectorizer'];
    label_encoder_y = trained_data['label_encoder_y'];
    
    x_test = bow(vectorizer,cleaned_query);
    
    #encode and predict
    intent = predict(classifier,label_encoder_y,x_test);
    response_voice = response('intents.json',intent);
    print('response : ' , response_voice)
    espeak = 'espeak "'+ response_voice + '" 2>/dev/null';
    print(espeak)
    call([espeak], shell=True)