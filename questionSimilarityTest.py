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


nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

#print(doc2vec.FAST_VERSION)

df = pd.read_csv('questionsData.csv', encoding = "ISO-8859-1", usecols=['Id', 'Title', 'Body', 'Tags'])

df.shape

#%%
questions1 = (df['Title']).values.tolist() #d2v.model
questions = (df['Title'] +"."+ df['Body']).values.tolist() #d2vII.model

tagged_data = [TaggedDocument(words=word_tokenize(question.lower()), tags=[str(i)]) for i, question in enumerate(questions1)]

# ids = df['Id'].values.tolist()
# tagged_data = [TaggedDocument(words=question, tags=[ids[i]]) for i, question enumerate(questions)]

#print("Processed Questions: \n(1) - {0}\n(2) - {1}\n(3) - {2}".format(processedQuestions[0], processedQuestions[1], processedQuestions[2]))
#questions[2] 

#%%
max_epochs = 50
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

model.save("d2v.model")
print("Model Saved")

#%%
model = Doc2Vec.load("d2v.model")
model2 = Doc2Vec.load("d2vII.model")
#to find the vector of a document which is not in training data
test_data = word_tokenize("I want to know how to delete a character from a string".lower())
v1 = model.infer_vector(test_data)
v2 = model2.infer_vector(test_data)
print(model.docvecs.most_similar([v1]))
print("")
print(model2.docvecs.most_similar([v2]))

#%%
# test_data = [w for w in word_tokenize("how exactly can i go into a string and remove a character".lower()) if w not in stop_words]
# test_data2 = [w for w in word_tokenize("How to delete a character from a string using python?".lower()) if w not in stop_words]
# test_data3 = [w for w in word_tokenize("Remove or Replace a character in a string".lower()) if w not in stop_words]

test_data = word_tokenize("how can i remove a character from a string".lower())
test_data2 = word_tokenize("How to delete a character from a string using python?".lower())
test_data3 = word_tokenize("Remove or Replace a character in a string".lower())

v1 = model.infer_vector(test_data)
v2 = model.infer_vector(test_data2)
v3 = model.infer_vector(test_data3)

print(model.wv.n_similarity(test_data2,test_data))
print(model.wv.n_similarity(test_data3,test_data))

#model.wv.most_similar("list")
#print(questions1[16195])

#print("V1_infer", v1)

# to find most similar doc using tags
#similar_doc = model.docvecs.most_similar('1')
#print(similar_doc)


# to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
#print(model.docvecs['1'])