# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 22:28:05 2017

@author: arunk
"""
import json;
import re;
from nltk.stem.lancaster import LancasterStemmer;
from nltk.corpus import stopwords;
import pickle;

from sklearn.preprocessing import LabelEncoder;
from sklearn.naive_bayes import GaussianNB,MultinomialNB;
from sklearn.ensemble import VotingClassifier;

#cleaning the texts
def clean_texts(dataset):
    corpus = [];
    #retain only alphabets
    for i in range (0 , len(dataset)): 
        review = re.sub("[^a-z A-Z]", " ", dataset[i]);
        review = review.lower();
        clean_text = review.split();
        ls = LancasterStemmer();
        review = [ls.stem(word) for word in clean_text if word not in set(stopwords.words("english"))];         
        if(len(review) < 2):
            clean_text = [ls.stem(word) for word in clean_text];
        else:
            clean_text = review;
        
        clean_text = " ".join(clean_text);    
        corpus.append(clean_text);
    return corpus;

#create bag of words model using vectorizer
def bow(corpus):
    from sklearn.feature_extraction.text import CountVectorizer;
    vectorizer = CountVectorizer();
    vectorizer.fit(corpus);
    
    x_train = vectorizer.transform(corpus).toarray();
    
    return x_train,vectorizer;

#fit classifier for x_train and y_train
#fit y_train by encoding it
def fit_classifier(x_train,y_train):
    label_encoder_y = LabelEncoder();
    y_train = label_encoder_y.fit_transform(y_train);

    #create classifier by combining GAussian and Multinomial naive bayes
    clf1 = MultinomialNB();
    clf2 = GaussianNB();
    classifier = VotingClassifier(estimators=[('mnb',clf1),('gnb',clf2)],
                                              voting='soft',
                                              weights=[1,3]);
    classifier.fit(x_train,y_train);
    return classifier,label_encoder_y;                                              
    
    
def train_data(intent_file,train_file):
    with open(intent_file,'r') as json_data:
        dataset = json.load(json_data);
        
    queries = [];
    y_train = [];
    
    #clean training texts in corpus
    #loop through each sentence in our intents patterns
    for train_data in dataset['train_set']:
        for pattern in train_data['patterns']:
            queries.append(pattern);
            y_train.append(train_data['intent']);
            
    corpus = clean_texts(queries);
    
    x_train,vectorizer = bow(corpus);
    classifier,label_encoder_y = fit_classifier(x_train,y_train);
    
    #store trained data
    pickle.dump({'classifier':classifier,'vectorizer':vectorizer,'label_encoder_y':label_encoder_y},
                open(train_file,'wb'));
                
train_data('intents.json','training_data.pkl');
    