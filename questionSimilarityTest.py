#%% [markdown]
#The Start Of Something New

#%%
import pandas as pd
import numpy as np 
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import datetime


#nltk.download('punkt')
#nltk.download('stopwords')
stop_words = stopwords.words('english')

#print(doc2vec.FAST_VERSION)

df = pd.read_csv('questionsData.csv', encoding = "ISO-8859-1", usecols=['Id', 'Title', 'Body', 'Tags'])

df.shape

#%%
#questions = (df['Title']).values.tolist() #d2v.model
questions = (df['Title'] +"."+ df['Body']).values.tolist() #d2vII.model

tagged_data = [TaggedDocument(words=word_tokenize(question.lower()), tags=[str(i)]) for i, question in enumerate(questions)]

#def preProcess(question):
#    wtokens = word_tokenize(question)
#    return [w for w in wtokens if w not in stop_words]

#processedQuestions = [preProcess(question) for question in questions]
#print("Processed Questions: \n(1) - {0}\n(2) - {1}\n(3) - {2}".format(processedQuestions[0], processedQuestions[1], processedQuestions[2]))
#questions[2] 

#%%
max_epochs = 20
vec_size = 150
alpha = 0.025

model = Doc2Vec(vector_size=vec_size, alpha=alpha, min_alpha=0.00025, window = 10, min_count=10, dm =1, workers=4)
  
model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0} at {1}'.format(epoch, datetime.datetime.now().time()))
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2vII.model")
print("Model Saved")

#%%
model= Doc2Vec.load("d2vII.model")
#to find the vector of a document which is not in training data
test_data = word_tokenize("how do i get an element from a list".lower())
v1 = model.infer_vector(test_data)
print(model.docvecs.most_similar([v1]))
#print("V1_infer", v1)

# to find most similar doc using tags
#similar_doc = model.docvecs.most_similar('1')
#print(similar_doc)


# to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
#print(model.docvecs['1'])