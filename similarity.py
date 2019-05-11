#%%
import pandas as pd
import numpy as np 
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import heapq
from operator import itemgetter

# I had to download these in order to use the stop words
# I believe one time download is enough
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Had to remove these words from the set because they are also python key words
stop_words.discard("for")
stop_words.discard("in")
stop_words.discard("while")
stop_words.discard("not")
stop_words.discard("if")

#%%
# NOTE: VERY IMPORTANT: load the model file
model = Doc2Vec.load("d2v.model")

# function to be used on all questions before processing
def preProcess(sentence):
    tokens = word_tokenize(sentence) 
    words = [w for w in tokens if ((w not in stop_words) and (w.isalpha()))]
    return words

def prepare4DB(question):
    return (question, model.infer_vector( preProcess(question), epochs=500))

# NOTE: The below n_most_similar function is faster than the n_most_similar2 function that comes after
# user_q - inputed question (*string) asked by user
# db_qs - tuple of two lists: *1st list is a list of questions (*strings)
#                             *2nd list is the list of infered vectors for each question (* numpy array of floats)
#         NOTE: Vector at index i in 2nd list, is the infered vector for Question at index i in the 1st list
# n - the number of most similar questions needed
def n_most_similar(user_q, db_qs, n):
    v1 = model.infer_vector( preProcess(user_q), epochs=500)
    print("Infered vectors for User-Questions ...")

    v2 = db_qs[1]
    v2 = np.array(v2)
    print("Infered vectors for DB-Questions ...")

    quest_values = [(db_qs[0][i], val) for i, val in enumerate(model.wv.cosine_similarities(v1, v2))]
    print("Calculated cosine similarity ...")

    n_questions = [x[0] for x in heapq.nlargest(n, quest_values, key=itemgetter(1))]
    print("Got similar questions ...")    

    return n_questions

# NOTE: The below n_most_similar2 function is slower than the above n_most_similar function because it
# it will infer the vector for all questions in the db_qs list
#
# user_q - inputed question (*string) asked by user
# db_qs - list of questions (*list of strings) obtained from database
# n - the number of most similar questions needed
def n_most_similar2(user_q, db_qs, n):
    v1 = model.infer_vector( preProcess(user_q), epochs=500)
    print("Infered vectors for User-Questions ...")

    v2 = [ model.infer_vector( preProcess(v), epochs=500) for v in db_qs]
    v2 = np.array(v2)
    print("Infered vectors for DB-Questions ...")

    quest_values = [(db_qs[i], val) for i, val in enumerate(model.wv.cosine_similarities(v1, v2))]
    print("Calculated cosine similarity ...")

    n_questions = [x[0] for x in heapq.nlargest(n, quest_values, key=itemgetter(1))]
    print("Got similar questions ...")    

    return n_questions


    