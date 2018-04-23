import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
tfidf_vectorizer = TfidfVectorizer()
import numpy as np

def who(string):    
    i = ["who",string]
    tf = tfidf_vectorizer.fit_transform(i)
    sim = cosine_similarity(tf[0:1],tf)
    return sim
def what(string):
    i = ["what",string]
    tf = tfidf_vectorizer.fit_transform(i)
    sim = cosine_similarity(tf[0:1],tf)
    return sim

def when(string):
    i = ["time",string]
    tf = tfidf_vectorizer.fit_transform(i)
    sim = cosine_similarity(tf[0:1],tf)
    return sim


def affirmation(string):
    i = ["is there it will can do does would could which has will",string]
    tf = tfidf_vectorizer.fit_transform(i)
    sim = cosine_similarity(tf[0:1],tf)
    return sim

def unknown(string):
    i = ["how there whose which where",string]
    tf = tfidf_vectorizer.fit_transform(i)
    sim = cosine_similarity(tf[0:1],tf)
    return sim



input_string = input("please enter the string:")

who_match = np.array(who(input_string)).tolist()[0][1]
what_match = np.array(what(input_string)).tolist()[0][1]
when_match = np.array(when(input_string)).tolist()[0][1]
affirmation_match = np.array(affirmation(input_string)).tolist()[0][1]
unknown_match = np.array(unknown(input_string)).tolist()[0][1]


if who_match >0 and what_match <= 0 and when_match <= 0:
    print("string belongs to 'who' class")
    
elif who_match <=0 and what_match > 0 and when_match <= 0:
    print("string belongs to 'what' class")
    
elif who_match <=0 and what_match > 0 and when_match > 0:
    print("string belongs to 'when' class")

elif who_match <=0 and what_match <= 0 and when_match <= 0 and affirmation_match > 0 and unknown_match < 0:
    print("string belongs to 'affirmation' class")

elif who_match <=0 and what_match <= 0 and when_match <= 0 and affirmation_match > 0 and unknown_match > 0:
    print("string belongs to 'unknown' class")


    
print(who_match,what_match,when_match,affirmation_match)









