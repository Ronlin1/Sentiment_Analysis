import pickle
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
import os

# download stop words
nltk.download('stopwords')

# Just for debugging:
cur_dir = os.path.dirname(__file__)
print(cur_dir)
# now we keep the stopwords here temporarily
stop = []


# Tokenising to remove stopwords, emojis etc
def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + \
           ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    # this will return the cleaned text
    return tokenized


# Conversion of text documents into matrix
vect = HashingVectorizer(decode_error='ignore',
                         n_features=2 ** 21,
                         preprocessor=None,
                         tokenizer=tokenizer)

# SGDClassifier is a linear classifier that uses SGD for training
clf = SGDClassifier(loss='log', random_state=1, max_iter=100, early_stopping=False)

# read data
df = pd.read_csv('movie_data.csv', encoding='utf-8')

# We fit the model with the reviews and labels of the dataset
X_train = df['review'].values
y_train = df['sentiment'].values

X_train = vect.transform(X_train)
clf.fit(X_train, y_train)


# creating the .pkl files
pickle.dump(stop,
            open('stopwords.pkl', 'wb'),
            protocol=4)

pickle.dump(clf,
            open('classifier.pkl', 'wb'),
            protocol=4)
