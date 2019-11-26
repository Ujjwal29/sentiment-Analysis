import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
from gensim.models import Word2Vec
from gensim.utils import tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

#TBD
'''
1. Remove stop words
2. Stemming
3. Lemmetization
4. better word embedding models like word2vec, glove
5. better ML classification algos
6. try deep learning algos like ANN
'''

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

df['tweet'][1] = [word for word in df['tweet'][1] if word not in stop_words]

df['tweet'] = df['tweet'].apply(lambda x : w.lower() for w in x)

combi = train.append(test, ignore_index=True)

#removing @ sign
combi['tidy_tweet'] = combi['tweet']
combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("@[\w]*", '')

#removing words with length < 3
combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x : ' '.join([w for w in x.split() if len(w) > 3]))

tokenized_tweet = combi['tidy_tweet'].apply(lambda x : x.split())

# ### Stemming : Stemming is a rule-based process of stripping the suffixes (“ing”, “ly”, “es”, “s” etc) from a word. For example, For example – “play”, “player”, “played”, “plays” and “playing” are the different variations of the word – “play”.

stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x : [stemmer.stem(i) for i in x])

#Stiching these tokens together
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
combi['tidy_tweet'] = tokenized_tweet

#WordCloud - all tweets
all_tweets = ' '.join([tweet for tweet in tokenized_tweet])
wc = WordCloud(width=800, height=500, random_state=21, max_font_size=100).generate(all_tweets)

plt.figure(figsize=(15,8))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

#WordCloud - positive tweets
all_positive_tweets = ''.join(t for t in combi['tidy_tweet'][combi['label']==0])

wc = WordCloud(width=800, height=500, max_font_size=100).generate(all_positive_tweets)

plt.figure(figsize=(15,8))
plt.imshow(wc)
plt.axis('off')
plt.show()

#WordCloud - negative tweets
all_negative_tweets = ' '.join([t for t in combi['tidy_tweet'][combi['label']==1]])
wc = WordCloud(width=800, height=500, max_font_size=100, random_state=21).generate(all_negative_tweets)

plt.figure(figsize=(15,8))
plt.imshow(wc, interpolation='nearest')
plt.axis('off')
plt.show()

## Extracting hashtags
def extract_hashtag(tweets):
    hashtags=[]
    pat = r"#[\w]+"
    for tweet in tweets:
        ht = re.findall(pat, tweet)
        hashtags.append(ht)
    return hashtags

positive_ht = extract_hashtag(combi['tidy_tweet'][combi['label']==0])
positive_ht

#Even cooler way to flat a list
pos_h = sum(positive_ht,[])
pos_h


# ## Cool way to flat list in python

#Positive Hashtags
positive_hashtags = []
for ph in positive_ht:
    positive_hashtags.extend(ph)
positive_hashtags


#Negative hashtags

neg_hash = extract_hashtag(combi['tidy_tweet'][combi['label']==1])

negative_hashtags = []
for nh in neg_hash:
    negative_hashtags.extend(nh)
negative_hashtags


# ### WordCloud for negative and postive hashtags
# ### Positive Hashtags wordCloud

lis = [t.split('#') for t in positive_hashtags]
lis = [t[1] for t in lis]
lis = ' '.join(lis)

wc = WordCloud(width=800, height=500, max_font_size=110, random_state=21).generate(lis)

plt.figure(figsize=(15,12))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

# ### Negative hashtags WordCloud

lis = [t.split('#') for t in negative_hashtags]
lis = [t[1] for t in lis]
lis = ' '.join(lis)

wc = WordCloud(width=800, height=500, max_font_size=150, random_state=21, collocations = False).generate(lis)

plt.figure(figsize=(15,12))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

negative_hashtags.count('#sony')


# ## getting frequency of words (in form of a dict) of a list of words using NLTK

# ### Selecting top 10 (n) negative hashtags

a = nltk.FreqDist(negative_hashtags)

d = pd.DataFrame({'Hashtags': list(a.keys()), 'Count' : list(a.values())})

d = d.nlargest(columns='Count', n=10)

plt.figure(figsize=(12,6))
plt.bar(d['Hashtags'], height=d['Count'])
plt.show()


# ### Selecting top 10 (n) negative hashtags
b = nltk.FreqDist(positive_hashtags)
d = pd.DataFrame({'hashtags' : list(b.keys()), 'Count':list(b.values())})
d = d.nlargest(columns='Count', n=10)

plt.figure(figsize=(12,6))
plt.bar(d['hashtags'], d['Count'])
plt.show()


# ## Extracting Features from the cleaned tweets

# ## 1. BAG OF WORDS Approach using CountVectorizer

train_data = combi[combi['label'].isnull()==False]
test_data = combi[combi['label'].isnull()==True]


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_df=0.9, min_df=2, max_features=1000, stop_words='english')
bow_model = vectorizer.fit_transform(train_data['tidy_tweet'])
bow_model_test = vectorizer.fit_transform(test_data['tidy_tweet'])

bow_model.shape


# ## 2. TF-IDF approach

from sklearn.feature_extraction.text import TfidfVectorizer

tf_vec = TfidfVectorizer(max_df=0.9, min_df=2, max_features=1000, stop_words='english')
tf_model = tf_vec.fit_transform(train_data['tidy_tweet'])

tf_model.shape


# ## Applying ML models to train models and predict sentiment

# ## 1. Logistic regression
# 
# ### a. On Bow model

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

x_train_bow, xvalid_bow, y_train, yvalid = train_test_split(bow_model, train_data['label'], random_state=4, test_size=0.2)

lreg = LogisticRegression()
lreg.fit(x_train_bow, y_train)

predictions = lreg.predict(xvalid_bow)
print(f1_score(yvalid, predictions))

# Predciting on the test set

test_pred = lreg.predict(bow_model_test)
test['label'] = test_pred
submission = test[['id', 'label']]
submission.to_csv('submission_bow.csv', index=False)

# ### b. On Tf-IDF model

x_train_tf, xvalid_tf, y_train_tf, yvalid_tf = train_test_split(tf_model, train_data['label'], random_state=4, test_size=0.2)

lreg = LogisticRegression()
lreg.fit(x_train_tf, y_train_tf)

predictions = lreg.predict(xvalid_tf)
print(f1_score(yvalid_tf, predictions))

# Predciting on the test set

test_pred = lreg.predict(bow_model_test)
test['label'] = test_pred
submission = test[['id', 'label']]
submission.to_csv('submission_tf.csv', index=False)