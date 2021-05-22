import pickle
import os
import numpy as np
from vectoriser import vect

# we load the classifier.pkl we created before
clf = pickle.load(open('classifier.pkl', 'rb'))


def classify(document):
    # this dictionary returns as outputs 'negative ' or 'positive' instead of 0 or 1
    label = {0: 'negative', 1: 'positive'}
    # transforming the document into analyzable data
    X = vect.transform([document])
    y = clf.predict(X)[0]
    proba = np.max(clf.predict_proba(X))
    return label[y], proba


if __name__ == '__main__':
    prediction, probability = classify('i liked the movie')
    print("Prediction : " + prediction)
    print("Probability : " + str(round(probability * 100, 1)) + ' %')
