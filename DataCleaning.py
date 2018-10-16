# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 12:47:50 2018

@author: trevo
"""

# This code is designed to give us an understanding of what our data set looks like
# Just change the file name and run it chuck my chuck, chucks separated by #

import numpy as np
import pandas as pd
import time
from nltk.corpus import stopwords as nltk
import nltk
f = open('C:\\Users\\Debosmita\\Documents\\TextMining\\testfile_2018-01.txt','r+')

col = ['title','date']

#read file and copy it to a dataframe
df = pd.DataFrame(columns=col)
for line in f:
    title = line[:-12]
    created_utc = line[-11:]
    if ("{" in title) or ("crosspost_parent" in title):
        continue
    converted_utc = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(created_utc)))
    df2 = pd.DataFrame([[title, converted_utc]], columns=['title','date'])
    df = df.append(df2,ignore_index=True)  
    print("title",title, "created_utc", converted_utc)


# 2.2 Removing Punctuation
df['title'] = df['title'].str.replace('[^\\\w\s]','')

# copying dataframe to a new one so that the original df remains intact.
test_df = df

# 2.3 Removal of Stop Words (Issues skip for now)
from nltk.corpus import stopwords
stop = stopwords.words('english')
test_df['title'] = test_df['title'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
test_df['title'].head()


# 2.4 Common word removal
freq = pd.Series(' '.join(test_df['title']).split()).value_counts()[:10]
freq

freq = list(freq.index)
test_df['title'] = test_df['title'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
test_df['title'].head()

# 2.5 Rare words removal
freq = pd.Series(' '.join(test_df['title']).split()).value_counts()[-50:]
freq

freq = list(freq.index)
test_df['title'] = test_df['title'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
test_df['title'].head()
test_df['title'] = test_df['title'].apply(lambda x: " ".join(x for x in x.split() if x not in '\\u'))
test_df['title'].head()
test_df['title'].decode('utf-8')

def clean_text(row):
    # return the list of decoded cell in the Series instead 
    return [r.decode('unicode_escape').encode('ascii', 'ignore') for r in row]

# remove unicodes
test_df['title'] = test_df['title'].apply(lambda x: " ".join(x for x in x.split() if '\\u' not in x))

test_df['title'] = test_df['title'].apply(lambda x: " ".join(x for x in x.split() if '/u' not in x))

test_df['title'] = test_df['title'].apply(lambda x: " ".join(x for x in x.split() if '/' not in x))


test_df['title'] = test_df['title'].apply(lambda x: " ".join(x for x in x.split() if '\\' not in x))





#
#
## Names for Columns
#Columns = ['created_utc', 
#           'score',
#           'author',
#           'body',
#           'parent_id',
#           'something', 
#           'name',
#           'id']
#
## Reading in our SQL to CSV bitcoin data for May 2015, column names defined above
##train = pd.read_csv('C:/Users/Debosmita/Documents/TextMining/Bitcoin2015.csv',  encoding = "ISO-8859-1", names = Columns)
#
## Getting rid of missing values in body (literally only one value was messing it up so not worth reporting probably just an import error on my part)
#train = train[train['body'].notnull()]
#
## Checking our data visually by printing the first few values
#print(train.head())
#
## Renaming the columns if you want, just get rid of the # below and change the name of the df to train
## df.columns = ['a', 'b','c','d','e','f','g','h']
#
## 1.1 Number of Words: Getting the word count
#train['word_count'] = train['body'].apply(lambda x: len(str(x).split(" ")))
#train[['body','word_count']].head()
#
## 1.2 Number of characters
#train['char_count'] = train['body'].str.len() ## this also includes spaces
#train[['body','char_count']].head()
#
## 1.3 Average Word Length
#def avg_word(sentence):
#  words = sentence.split()
#  return (sum(len(word) for word in words)/len(words))
#
#train['avg_word'] = train['body'].apply(lambda x: avg_word(x))
#train[['body','avg_word']].head()
#
## 1.4 Number of stopwords: must import library below, delete #
## You may also need to pip or conda install the library
## You can do this by running your conda comand prompt in admin mod (right click, then 'run as admin')
## then simply type conda install nltk and press enter, it will as if you are ready to proceed, type y and press enter
## or use pip install nltk if you are in regular command prompt
#
## Its not working for us rn so just move on to the next one
#
#from nltk.corpus import stopwords as nltk
#import nltk
##nltk.download('stopwords')
#
#stop = stop_words = nltk.corpus.stopwords.words( 'english' )#stopwords.words('english')
#
#train['stopwords'] = train['body'].apply(lambda x: len([x for x in x.split() if x in stop]))
#train[['body','stopwords']].head()
#
## 1.5 Number of special characters
#train['hastags'] = train['body'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
#train[['body','hastags']].head()
#
## 1.6 Number of numerics
#train['numerics'] = train['body'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
#train[['body','numerics']].head()
#
## 1.7 Number of Uppercase words
#train['upper'] = train['body'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
#train[['body','upper']].head()
#
######################################################################################################
## 2. Basic Pre-processing
#
## 2.1 Lower case: converting all terms to lower case
#train['body'] = train['body'].apply(lambda x: " ".join(x.lower() for x in x.split()))
#train['body'].head()
#
## 2.2 Removing Punctuation
#train['body'] = train['body'].str.replace('[^\w\s]','')
#train['body'].head()
#
## 2.3 Removal of Stop Words (Issues skip for now)
#from nltk.corpus import stopwords
#stop = stopwords.words('english')
#train['body'] = train['body'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
#train['body'].head()
#
## 2.4 Common word removal
#freq = pd.Series(' '.join(train['body']).split()).value_counts()[:10]
#freq
#
#freq = list(freq.index)
#train['body'] = train['body'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
#train['body'].head()
#
## 2.5 Rare words removal
#freq = pd.Series(' '.join(train['body']).split()).value_counts()[-10:]
#
#freq = list(freq.index)
#train['body'] = train['body'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
#train['body'].head()
#
## 2.6 Spelling correction
## library textblob as explained in 1.4 # pip install -U textblob
#from textblob import TextBlob
#train['body'][:5].apply(lambda x: str(TextBlob(x).correct()))
#
## 2.7 Tokenization
#TextBlob(train['body'][1]).words
#
#nltk.download('punkt')
## 2.8 Stemming
#from nltk.stem import PorterStemmer
#st = PorterStemmer()
#
#train['body'][:5].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
#    
#    # 2.9 Lemmatization (This takes a minute)
#    from textblob import Word
#    
#    train['body'] = train['body'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
#    train['body'].head()
#
######################################################################################################
## 3. Advance Text Processing
#
## 3.1 N-grams
#TextBlob(train['body'][0]).ngrams(2)
#
## 3.2 Term Frequency
#tf1 = (train['body'][1:2]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
#tf1.columns = ['words','tf']
#tf1
#
## 3.3 Inverse Document Frequency
#for i,word in enumerate(tf1['words']):
#  tf1.loc[i, 'idf'] = np.log(train.shape[0]/(len(train[train['body'].str.contains(word)])))
#tf1
#
## 3.4 Term Frequency – Inverse Document Frequency (TF-IDF)
#from sklearn.feature_extraction.text import TfidfVectorizer
#
#tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',stop_words= 'english',ngram_range=(1,1))
#train_vect = tfidf.fit_transform(train['body'])
#
#train_vect
#
## 3.5 Bag of Words
#from sklearn.feature_extraction.text import CountVectorizer
#
#bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")
#train_bow = bow.fit_transform(train['body'])
#train_bow
#                        
## 3.6 Sentiment Analysis ( This takes quite a while )
#train['sentiment'] = train['body'].apply(lambda x: TextBlob(x).sentiment[0] )
#train[['body','sentiment']].head()
#
#### 3.7 Word Embeddings
#from gensim.scripts.glove2word2vec import glove2word2vec
#
#glove_input_file = 'glove.6B.100d.txt'
#word2vec_output_file = 'glove.6B.100d.txt.word2vec'
#glove2word2vec(glove_input_file, word2vec_output_file)
#
## compacting the file to print to csv
#train_clean = train.drop(['parent_id','something', 'name','id'], axis=1)
#
## Printing our modified data to CSV
#train_clean.to_csv('CleanMay2015.csv')
#
#
#
#
#
#
#
#                        