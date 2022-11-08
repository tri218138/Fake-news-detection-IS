import pickle
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(
        word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

model = pickle.load(open('logreg_fakenews.pickle', 'rb'))
vectorizer = pickle.load(open('tfidf.pickle', 'rb'))

input = [stemming(x) for x in ["You are not true"]]
vec_input = vectorizer.transform(input).toarray()
is_fake_news = model.predict(vec_input)

print(is_fake_news)
