#!/usr/bin/env python
# coding: utf-8

# ## Problem Statement : given the tweets from customers about various tech firms who manufacture and sell mobiles, computers, laptops, etc, the task is to identify if the tweets have a negative sentiment towards such companies or products.

import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
from gensim.models import Word2Vec
from gensim.utils import tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn
from nltk.stem import WordNetLemmatizer, PorterStemmer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('sentiwordnet')
nltk.download('wordnet')
from nltk.stem.porter import *
import re

def remove_pattern(text, pattern):
    r = re.findall(pattern, text)
    print(r)
    for i in r:
        text = r.sub(i,'', text)
    return text

stop_words = set(stopwords.words('english'))

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train['tweet'] = train['tweet'].apply(lambda x : x.split())

df['tweet'] = df['tweet'].apply(lambda x : w.lower() for w in x)

combi = train.append(test, ignore_index=True)
combi['tidy_tweet'] = combi['tweet']
combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("@[\w]*", '')

#removing words with length < 3
combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x : ' '.join([w for w in x.split() if len(w) > 3]))

tokenized_tweet = combi['tidy_tweet'].apply(lambda x : x.split())

stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x : [stemmer.stem(i) for i in x])

for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
combi['tidy_tweet'] = tokenized_tweet

#removing hashtags symbol
tokenized_tweet = combi['tidy_tweet']

#removing hashtags and web addresses
tokenized_tweet = [t.split() for t in tokenized_tweet]

for tweet in tokenized_tweet:
    for i in range(len(tweet)):
        txt = tweet[i]
        result = re.match(r'#+', tweet[i])
        if result:
            tweet[i] = tweet[i].split('#')[1]
        tweet[i] = re.sub('https?://[A-Za-z0-9./]+','', tweet[i])

tokenized_tweet = [' '.join(t) for t in tokenized_tweet]

combi['tidy_tweet'] = tokenized_tweet

## --Running Pos_tag and neg_tag function

df_copy = combi.copy()
pstem = PorterStemmer()
lem = WordNetLemmatizer()

def pos_senti(df_copy):#takes
    li_swn=[]
    li_swn_pos=[]
    li_swn_neg=[]
    missing_words=[]
    for i in range(len(df_copy.index)):
        text = df_copy.loc[i]['tidy_tweet']
        tokens = nltk.word_tokenize(text)
        tagged_sent = nltk.pos_tag(tokens)
        store_it = [(word, nltk.map_tag('en-ptb', 'universal', tag)) for word, tag in tagged_sent]
        #print("Tagged Parts of Speech:",store_it)

        pos_total=0
        neg_total=0
        for word,tag in store_it:
            if(tag=='NOUN'):
                tag='n'
            elif(tag=='VERB'):
                tag='v'
            elif(tag=='ADJ'):
                tag='a'
            elif(tag=='ADV'):
                tag = 'r'
            else:
                tag='nothing'

            if(tag!='nothing'):
                concat = word+'.'+tag+'.01'
                try:
                    this_word_pos=swn.senti_synset(concat).pos_score()
                    this_word_neg=swn.senti_synset(concat).neg_score()
                    #print(word,tag,':',this_word_pos,this_word_neg)
                except Exception as e:
                    wor = lem.lemmatize(word)
                    concat = wor+'.'+tag+'.01'
                    # Checking if there's a possiblity of lemmatized word be accepted into SWN corpus
                    try:
                        this_word_pos=swn.senti_synset(concat).pos_score()
                        this_word_neg=swn.senti_synset(concat).neg_score()
                    except Exception as e:
                        wor = pstem.stem(word)
                        concat = wor+'.'+tag+'.01'
                        # Checking if there's a possiblity of lemmatized word be accepted
                        try:
                            this_word_pos=swn.senti_synset(concat).pos_score()
                            this_word_neg=swn.senti_synset(concat).neg_score()
                        except:
                            missing_words.append(word)
                            continue
                pos_total+=this_word_pos
                neg_total+=this_word_neg
        li_swn_pos.append(pos_total)
        li_swn_neg.append(neg_total)

        if(pos_total!=0 or neg_total!=0):
            if(pos_total>neg_total):
                li_swn.append(1)
            else:
                li_swn.append(-1)
        else:
            li_swn.append(0)
    df_copy.insert(4,"pos_score",li_swn_pos,True)
    df_copy.insert(5,"neg_score",li_swn_neg,True)
    df_copy.insert(6,"sent_score",li_swn,True)
    return df_copy

df = pos_senti(combi)

df['sent_score'] = df['sent_score'].apply(lambda x: x if x>0 else 0)

test_df = df[['id','sent_score']]
test_df = test_df[7920:]
test_df.rename({'sent_score': 'label'}, inplace=True)
test_df.to_csv('basic_model.csv', index=False)